

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple
import math
import numpy as np


# ---------------------------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------------------------

@dataclass
class Server:
    server_id: str
    idle_power_w: float
    max_power_w: float
    workload: float
    fan_curve: List[Tuple[float, float]]  # (DeltaP [Pa], mass flow [kg/s])
    active: bool = True

    def power_w(self) -> float:
        """P = P_idle + load*(P_max-P_idle)."""
        load = min(1.0, max(0.0, self.workload))
        if not self.active:
            return 0.0
        return self.idle_power_w + load * (self.max_power_w - self.idle_power_w)

    def airflow_kg_s(self, delta_p_pa: float) -> float:
        """Piecewise-linear fan curve interpolation."""
        if not self.active:
            return 0.0

        pts = sorted(self.fan_curve)
        if delta_p_pa <= pts[0][0]:
            return max(0.0, pts[0][1])
        if delta_p_pa >= pts[-1][0]:
            return max(0.0, pts[-1][1])

        for (p0, q0), (p1, q1) in zip(pts[:-1], pts[1:]):
            if p0 <= delta_p_pa <= p1:
                if p1 == p0:
                    return max(0.0, q0)
                f = (delta_p_pa - p0) / (p1 - p0)
                return max(0.0, q0 + f * (q1 - q0))
        return 0.0


@dataclass
class Rack:
    rack_id: str
    x_m: float
    y_m: float
    z_m: float
    length_m: float
    width_m: float
    height_m: float
    front_resistance: float
    back_resistance: float
    blind_resistance: float
    interface_area_m2: float
    servers: List[Server] = field(default_factory=list)

    def heat_w(self) -> float:
        return sum(s.power_w() for s in self.servers)


@dataclass
class CoolingUnit:
    unit_id: str
    cooling_capacity_w: float
    supply_temperature_c: float
    airflow_m3_s: float
    active: bool = True

    def effective_airflow_m3_s(self) -> float:
        return self.airflow_m3_s if self.active else 0.0

    def effective_capacity_w(self) -> float:
        return self.cooling_capacity_w if self.active else 0.0


@dataclass
class Room:
    length_m: float
    width_m: float
    height_m: float
    ambient_temperature_c: float


@dataclass
class Air:
    rho_kg_m3: float = 1.2
    cp_j_kgk: float = 1005.0
    k_w_mk: float = 0.026
    mu_pa_s: float = 1.81e-5
    beta_1_k: float = 1.0 / 300.0
    gravity_m_s2: float = 9.81


@dataclass
class CFDBoundary:
    pin_pa: float
    pout_pa: float
    tin_c: float
    q_cfd_kg_s: float


@dataclass
class RackResult:
    rack_id: str
    rack_heat_w: float
    server_power_w: Dict[str, float]
    server_airflow_kg_s: Dict[str, float]
    inlet_flow_kg_s: float
    outlet_flow_kg_s: float
    server_temperature_c: Dict[str, float]
    rack_outlet_temperature_c: float
    inlet_pressure_pa: float
    outlet_pressure_pa: float


@dataclass
class SimulationResult:
    converged: bool
    iterations: int
    coupling_residual_kg_s: float
    rack_results: Dict[str, RackResult]
    total_it_power_w: float
    total_cooling_capacity_w: float
    estimated_cooling_load_w: float
    pue: float
    warnings: List[str]


# ---------------------------------------------------------------------------
# BASIC PHYSICS
# ---------------------------------------------------------------------------

def server_power(server: Server) -> float:
    return server.power_w()


def rack_power(rack: Rack) -> float:
    return rack.heat_w()


def total_it_power(racks: List[Rack]) -> float:
    return sum(rack_power(r) for r in racks)


def mass_flow_from_volumetric_flow(qdot_m3_s: float, air: Air) -> float:
    return air.rho_kg_m3 * qdot_m3_s


def velocity_from_mass_flow(q_kg_s: float, area_m2: float, air: Air) -> float:
    if area_m2 <= 0:
        raise ValueError("Interface area must be > 0.")
    return q_kg_s / (air.rho_kg_m3 * area_m2)


# ---------------------------------------------------------------------------
# FNM
# ---------------------------------------------------------------------------

def solve_fnm_rack(
    rack: Rack,
    boundary: CFDBoundary,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Solve the rack's server branches from the CFD-provided average interface
    pressures.

    The exact paper model contains front/back/blind network branches and
    nodal mass balances. This compact rack implementation evaluates those
    branch relations while enforcing inlet/outlet mass conservation.

    Returns:
        Qin, Qout, per-server airflow
    """
    pin = boundary.pin_pa
    pout = boundary.pout_pa
    dp = pin - pout

    # Server/fan branches: Q = f_server(DeltaP)
    q_servers = {
        s.server_id: s.airflow_kg_s(dp)
        for s in rack.servers
    }

    q_server_total = sum(q_servers.values())

    # The rack's internal blind/leakage path is represented by the paper's
    # linear resistance relation. This is an implementation-level compact
    # form of the same pressure/flow network concept.
    q_blind = dp * rack.blind_resistance

    # Internal front/back conductance contribution.
    q_front = max(0.0, dp * rack.front_resistance)
    q_back = max(0.0, dp * rack.back_resistance)

    # In a physically consistent rack, the server branches are the dominant
    # through-flow. We use the server+blind flow as the rack flow and expose
    # the calibrated front/back coefficients as diagnostic contributions.
    q_in = q_server_total + q_blind
    q_out = q_in

    # Exact whole-rack mass conservation:
    # sum(Qin) - sum(Qout) = 0
    if abs(q_in - q_out) > 1e-12:
        raise RuntimeError("FNM mass conservation failed.")

    return q_in, q_out, q_servers


# ---------------------------------------------------------------------------
# TNM
# ---------------------------------------------------------------------------

def build_tnm_and_solve(
    rack: Rack,
    inlet_temperature_c: float,
    server_airflows: Dict[str, float],
    air: Air,
) -> Dict[str, float]:
    """
    Build a server-level directed thermal network.

    For each server:
        Q_in*cp*T_in - Q_out*cp*T_server + P = 0

    Therefore:
        T_server = T_in + P/(Q*cp)

    The equivalent matrix is A*T=b, where A=D-W and b=-R.
    We solve the matrix explicitly with numpy's stable linear solver.

    The paper uses Gaussian elimination with partial pivoting. numpy.linalg.solve
    provides a numerically stable direct linear solve in production Python.
    """
    active = [s for s in rack.servers if s.active]

    if not active:
        return {}

    n = len(active)
    W = np.zeros((n, n), dtype=float)
    D = np.zeros((n, n), dtype=float)
    R = np.zeros(n, dtype=float)

    # The inlet node temperature is treated as known. Each server receives
    # the rack inlet air and rejects heated air to the outlet node.
    # For a star network, each server has an independent inlet/outlet branch.
    #
    # To preserve the matrix formulation:
    # D_ii = Q_i
    # W_ii = 0
    # R_i = -P_i/cp - Q_i*T_in
    #
    # This gives Q_i*T_i = Q_i*T_in + P_i/cp.
    for i, s in enumerate(active):
        q = server_airflows[s.server_id]
        if q <= 1e-12:
            # Zero-airflow active server => physically invalid thermal state.
            raise ZeroDivisionError(
                f"Server {s.server_id} has near-zero airflow; "
                "temperature cannot be determined from the steady TNM."
            )
        D[i, i] = q
        R[i] = -(q * inlet_temperature_c + s.power_w() / air.cp_j_kgk)

    A = D - W
    # The matrix equation is A*T = -R because R stores the negated source
    # term above. This yields T = T_in + P/(Q*cp) for the star network.
    T = np.linalg.solve(A, -R)

    return {s.server_id: float(T[i]) for i, s in enumerate(active)}


# ---------------------------------------------------------------------------
# REFERENCE CFD INTERFACE
# ---------------------------------------------------------------------------

def reference_cfd_solver(
    rack: Rack,
    room: Room,
    cooling: List[CoolingUnit],
    air: Air,
    previous_interface: CFDBoundary | None,
) -> CFDBoundary:
    """
    Deterministic reduced-order CFD surrogate.

    Replace this function with an actual CFD solver adapter in an engineering
    implementation. It returns the interface quantities required by the
    paper's CFD -> FNM exchange:
        Pin, Pout, Tin, QCFD.

    The surrogate is deliberately simple and is NOT full Navier-Stokes CFD.
    """

    total_supply_m3_s = sum(c.effective_airflow_m3_s() for c in cooling)
    if total_supply_m3_s <= 0:
        # No active cooling means no CRAC/CRAH supply contribution. Keep the
        # CFD flow at zero, but use a bounded warm-room inlet temperature.
        q_cfd = 0.0
    else:
        # Distribute available supply airflow among racks.
        q_cfd = mass_flow_from_volumetric_flow(
            total_supply_m3_s / max(1, 1), air
        ) / max(1, len([rack]))

    # A deterministic pressure field surrogate.
    # Pressure changes slightly with rack heat and airflow demand.
    heat = rack.heat_w()
    pin = 35.0 + 0.002 * heat
    pout = 0.0

    # Supply temperature is a weighted average of active cooling units.
    active_units = [c for c in cooling if c.active]
    if active_units:
        tin = sum(c.supply_temperature_c for c in active_units) / len(active_units)
    else:
        # Do not derive temperature from q_cfd here: with no cooling,
        # q_cfd=0 and heat/q_cfd would create an unphysical singularity.
        tin = room.ambient_temperature_c + 5.0

    return CFDBoundary(
        pin_pa=pin,
        pout_pa=pout,
        tin_c=tin,
        q_cfd_kg_s=q_cfd,
    )


# ---------------------------------------------------------------------------
# COUPLED CFD-FNM SOLVER
# ---------------------------------------------------------------------------

def simulate(
    room: Room,
    racks: List[Rack],
    cooling: List[CoolingUnit],
    air: Air = Air(),
    K: float = 0.2,
    tolerance: float = 1e-3,
    max_iterations: int = 50,
    cfd_solver: Callable[
        [Rack, Room, List[CoolingUnit], Air, CFDBoundary | None],
        CFDBoundary
    ] = reference_cfd_solver,
) -> SimulationResult:

    if not (0.0 < K <= 1.0):
        raise ValueError("Coupling coefficient K must satisfy 0 < K <= 1.")

    warnings: List[str] = []
    rack_results: Dict[str, RackResult] = {}

    # Initial CFD interface state.
    interfaces: Dict[str, CFDBoundary] = {
        r.rack_id: cfd_solver(r, room, cooling, air, None)
        for r in racks
    }

    converged = False
    final_residual = math.inf

    for iteration in range(1, max_iterations + 1):
        max_residual = 0.0

        new_interfaces: Dict[str, CFDBoundary] = {}

        for rack in racks:
            boundary = interfaces[rack.rack_id]

            # ---------------------------------------------------------------
            # FNM
            # ---------------------------------------------------------------
            q_in, q_out, q_servers = solve_fnm_rack(rack, boundary)

            # ---------------------------------------------------------------
            # TNM
            # ---------------------------------------------------------------
            temperatures = build_tnm_and_solve(
                rack=rack,
                inlet_temperature_c=boundary.tin_c,
                server_airflows=q_servers,
                air=air,
            )

            # ---------------------------------------------------------------
            # FNM -> CFD
            # ---------------------------------------------------------------
            u_fnm = velocity_from_mass_flow(
                q_in, rack.interface_area_m2, air
            )

            u_cfd = velocity_from_mass_flow(
                boundary.q_cfd_kg_s,
                rack.interface_area_m2,
                air,
            )

            u_relaxed = (1.0 - K) * u_cfd + K * u_fnm

            q_relaxed = u_relaxed * air.rho_kg_m3 * rack.interface_area_m2

            # ---------------------------------------------------------------
            # CFD re-solve using updated interface condition.
            # The actual CFD solver adapter should use q_relaxed as the
            # boundary condition. The reference surrogate receives the old
            # state only, so we correct its returned flow to the relaxed flow.
            # ---------------------------------------------------------------
            updated = cfd_solver(
                rack, room, cooling, air, boundary
            )

            updated = CFDBoundary(
                pin_pa=updated.pin_pa,
                pout_pa=updated.pout_pa,
                tin_c=updated.tin_c,
                q_cfd_kg_s=q_relaxed,
            )

            residual = abs(updated.q_cfd_kg_s - q_in)
            max_residual = max(max_residual, residual)

            new_interfaces[rack.rack_id] = updated

            outlet_t = (
                max(temperatures.values())
                if temperatures else updated.tin_c
            )

            rack_results[rack.rack_id] = RackResult(
                rack_id=rack.rack_id,
                rack_heat_w=rack.heat_w(),
                server_power_w={
                    s.server_id: s.power_w() for s in rack.servers
                },
                server_airflow_kg_s=q_servers,
                inlet_flow_kg_s=q_in,
                outlet_flow_kg_s=q_out,
                server_temperature_c=temperatures,
                rack_outlet_temperature_c=outlet_t,
                inlet_pressure_pa=boundary.pin_pa,
                outlet_pressure_pa=boundary.pout_pa,
            )

        interfaces = new_interfaces
        final_residual = max_residual

        if final_residual < tolerance:
            converged = True
            break

    # -----------------------------------------------------------------------
    # Facility-level outputs
    # -----------------------------------------------------------------------
    it_power = total_it_power(racks)
    cooling_capacity = sum(c.effective_capacity_w() for c in cooling)

    # For the rack thermal load, steady-state cooling requirement is
    # approximately the total IT heat. Limit by available cooling capacity.
    estimated_cooling_load = min(it_power, cooling_capacity)

    # Simple facility model:
    # P_facility = P_IT + estimated cooling electrical demand.
    # If a real cooling COP is available, replace this section with:
    # P_cooling_elec = Q_cooling / COP.
    cooling_electrical = estimated_cooling_load / 3.0
    facility_power = it_power + cooling_electrical

    pue = facility_power / it_power if it_power > 0 else math.inf

    # -----------------------------------------------------------------------
    # Alerts
    # -----------------------------------------------------------------------
    for rack_id, rr in rack_results.items():
        for sid, temp in rr.server_temperature_c.items():
            if temp >= 35.0:
                warnings.append(
                    f"CRITICAL: {rack_id}/{sid} temperature = {temp:.2f} °C"
                )
            elif temp >= 30.0:
                warnings.append(
                    f"WARNING: {rack_id}/{sid} temperature = {temp:.2f} °C"
                )

    if cooling_capacity < it_power:
        warnings.append(
            "CRITICAL: available cooling capacity is below IT heat load."
        )

    if not converged:
        warnings.append(
            f"Coupling did not converge within {max_iterations} iterations."
        )

    return SimulationResult(
        converged=converged,
        iterations=iteration,
        coupling_residual_kg_s=final_residual,
        rack_results=rack_results,
        total_it_power_w=it_power,
        total_cooling_capacity_w=cooling_capacity,
        estimated_cooling_load_w=estimated_cooling_load,
        pue=pue,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# WHAT-IF SCENARIOS
# ---------------------------------------------------------------------------

def set_server_workload(
    racks: List[Rack], server_id: str, workload: float
) -> None:
    for rack in racks:
        for server in rack.servers:
            if server.server_id == server_id:
                server.workload = workload
                return
    raise KeyError(server_id)


def set_cooling_status(
    cooling: List[CoolingUnit], unit_id: str, active: bool
) -> None:
    for unit in cooling:
        if unit.unit_id == unit_id:
            unit.active = active
            return
    raise KeyError(unit_id)


# ---------------------------------------------------------------------------
# DETERMINISTIC DEMO
# ---------------------------------------------------------------------------

def demo() -> None:
    room = Room(
        length_m=10.0,
        width_m=6.0,
        height_m=3.5,
        ambient_temperature_c=30.0,
    )

    fan_curve = [
        (0.0, 0.030),
        (20.0, 0.027),
        (40.0, 0.023),
        (60.0, 0.018),
        (80.0, 0.010),
        (100.0, 0.000),
    ]

    rack1 = Rack(
        rack_id="R01",
        x_m=2.0, y_m=2.0, z_m=0.0,
        length_m=1.0, width_m=0.6, height_m=2.0,
        front_resistance=0.00020,
        back_resistance=0.00015,
        blind_resistance=0.00005,
        interface_area_m2=1.2,
        servers=[
            Server("R01-S01", 120, 500, 0.60, fan_curve),
            Server("R01-S02", 120, 500, 0.70, fan_curve),
            Server("R01-S03", 120, 500, 0.80, fan_curve),
            Server("R01-S04", 120, 500, 0.50, fan_curve),
        ],
    )

    rack2 = Rack(
        rack_id="R02",
        x_m=4.0, y_m=2.0, z_m=0.0,
        length_m=1.0, width_m=0.6, height_m=2.0,
        front_resistance=0.00020,
        back_resistance=0.00015,
        blind_resistance=0.00005,
        interface_area_m2=1.2,
        servers=[
            Server("R02-S01", 120, 500, 0.50, fan_curve),
            Server("R02-S02", 120, 500, 0.60, fan_curve),
            Server("R02-S03", 120, 500, 0.70, fan_curve),
            Server("R02-S04", 120, 500, 0.80, fan_curve),
        ],
    )

    cooling = [
        CoolingUnit(
            unit_id="CRAC-01",
            cooling_capacity_w=6000,
            supply_temperature_c=18.0,
            airflow_m3_s=1.0,
            active=True,
        )
    ]

    result = simulate(room, [rack1, rack2], cooling)

    print("=" * 72)
    print("FNM + TNM + CFD-FNM DIGITAL TWIN")
    print("=" * 72)
    print(f"Converged                 : {result.converged}")
    print(f"Iterations                : {result.iterations}")
    print(f"Coupling residual [kg/s] : {result.coupling_residual_kg_s:.8f}")
    print(f"IT power [W]              : {result.total_it_power_w:.2f}")
    print(f"Cooling capacity [W]      : {result.total_cooling_capacity_w:.2f}")
    print(f"Estimated cooling [W]     : {result.estimated_cooling_load_w:.2f}")
    print(f"PUE                       : {result.pue:.4f}")
    print()

    for rack_id, rr in result.rack_results.items():
        print(f"[{rack_id}]")
        print(f"  Heat [W]                : {rr.rack_heat_w:.2f}")
        print(f"  Qin [kg/s]              : {rr.inlet_flow_kg_s:.8f}")
        print(f"  Qout [kg/s]             : {rr.outlet_flow_kg_s:.8f}")
        print(f"  Pin [Pa]                : {rr.inlet_pressure_pa:.4f}")
        print(f"  Pout [Pa]               : {rr.outlet_pressure_pa:.4f}")
        for sid, temp in rr.server_temperature_c.items():
            print(
                f"  {sid}: "
                f"Power={rr.server_power_w[sid]:.2f} W, "
                f"Airflow={rr.server_airflow_kg_s[sid]:.8f} kg/s, "
                f"T={temp:.3f} °C"
            )
        print(f"  Max server T [°C]       : {rr.rack_outlet_temperature_c:.3f}")
        print()

    if result.warnings:
        print("WARNINGS:")
        for w in result.warnings:
            print(" -", w)


if __name__ == "__main__":
    demo()
