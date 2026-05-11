"""ITTC-based water physics features for hull resistance modeling.

Implements ITTC Recommended Procedures 7.5-02-01-03 (Fresh Water and Seawater
Properties) to compute water density, kinematic viscosity, and frictional
resistance coefficient from sea surface temperature and salinity.

References:
    - ITTC 7.5-02-01-03 Rev 02 (2011): Fresh Water and Seawater Properties
    - ITTC 7.5-02-02-02 Rev 01 (2002): Resistance Uncertainty Analysis
    - Sharqawy et al. (2010): Thermophysical properties of seawater
    - TEOS-10 / IOC et al. (2010): Seawater equation of state

Constants:
    SALINITY_PASIG_RIVER: Estimated absolute salinity for the Pasig River
        ferry route (~10 g/kg, brackish water). Pasig River connects
        freshwater Laguna de Bay to seawater Manila Bay (~35 g/kg).
    VESSEL_LENGTH_M: Estimated waterline length of M/B Dalaray (~20 m,
        typical Philippine e-ferry catamaran).
"""

from __future__ import annotations

import math

# ── Constants ──
SALINITY_PASIG_RIVER = 10.0  # g/kg (absolute salinity, brackish estimate)
VESSEL_LENGTH_M = 20.0       # meters (estimated waterline length)


def seawater_density(temperature_c: float, salinity_g_kg: float = SALINITY_PASIG_RIVER) -> float:
    """Compute seawater density (kg/m³) using TEOS-10 polynomial approximation.

    Valid for 0-40degC and 0-42 g/kg salinity at standard pressure.
    Sharqawy et al. (2010), Eq. 8, simplified for standard pressure.

    Args:
        temperature_c: Sea surface temperature in degC.
        salinity_g_kg: Absolute salinity in g/kg.

    Returns:
        Density in kg/m³.
    """
    t = temperature_c
    s = salinity_g_kg

    # Fresh water density (IAPWS 2008a, polynomial fit 0-180degC)
    rho_fw = (
        999.842594
        + 6.793952e-2 * t
        - 9.095290e-3 * t**2
        + 1.001685e-4 * t**3
        - 1.120083e-6 * t**4
        + 6.536332e-9 * t**5
    )

    # Salinity correction (UNESCO/Millero 1981, simplified)
    if s > 0:
        rho_sw = rho_fw + s * (
            0.824493
            - 4.0899e-3 * t
            + 7.6438e-5 * t**2
            - 8.2467e-7 * t**3
            + 5.3875e-9 * t**4
        ) + s**1.5 * (
            -5.72466e-3
            + 1.0227e-4 * t
            - 1.6546e-6 * t**2
        ) + 4.8314e-4 * s**2
    else:
        rho_sw = rho_fw

    return rho_sw


def kinematic_viscosity(temperature_c: float, salinity_g_kg: float = SALINITY_PASIG_RIVER) -> float:
    """Compute kinematic viscosity (m²/s) using IAPWS/Sharqawy equations.

    Args:
        temperature_c: Sea surface temperature in degC.
        salinity_g_kg: Absolute salinity in g/kg.

    Returns:
        Kinematic viscosity in m²/s.
    """
    t = temperature_c

    # Fresh water absolute viscosity (IAPWS 2008a, Pa·s)
    # Simplified Vogel-type equation valid 0-180degC
    mu_fw = 4.2844e-5 + (0.157 * (t + 64.993)**2 - 91.296) ** (-1)

    # Seawater viscosity correction (Sharqawy et al. 2010, Eq. 22)
    s = salinity_g_kg
    if s > 0:
        A = 1.541 + 1.998e-2 * t - 9.52e-5 * t**2
        B = 7.974 - 7.561e-2 * t + 4.724e-4 * t**2
        mu_sw = mu_fw * (1 + A * (s / 1000) + B * (s / 1000)**2)
    else:
        mu_sw = mu_fw

    # Kinematic viscosity = absolute viscosity / density
    rho = seawater_density(temperature_c, salinity_g_kg)
    nu = mu_sw / rho

    return nu


def reynolds_number(speed_m_s: float, temperature_c: float,
                    length_m: float = VESSEL_LENGTH_M,
                    salinity_g_kg: float = SALINITY_PASIG_RIVER) -> float:
    """Compute Reynolds number for the vessel hull.

    Re = V * L / nu

    Args:
        speed_m_s: Vessel speed in m/s.
        temperature_c: Sea surface temperature in degC.
        length_m: Vessel waterline length in meters.
        salinity_g_kg: Absolute salinity in g/kg.

    Returns:
        Reynolds number (dimensionless). Returns 0 if speed is ~0.
    """
    if speed_m_s < 0.01:
        return 0.0

    nu = kinematic_viscosity(temperature_c, salinity_g_kg)
    return speed_m_s * length_m / nu


def friction_coefficient(speed_m_s: float, temperature_c: float,
                         length_m: float = VESSEL_LENGTH_M,
                         salinity_g_kg: float = SALINITY_PASIG_RIVER) -> float:
    """Compute ITTC-1957 frictional resistance coefficient.

    CF = 0.075 / (log10(Re) - 2)²

    This is the standard correlation line used worldwide for ship model-to-
    full-scale resistance prediction.

    Args:
        speed_m_s: Vessel speed in m/s.
        temperature_c: Sea surface temperature in degC.
        length_m: Vessel waterline length in meters.
        salinity_g_kg: Absolute salinity in g/kg.

    Returns:
        Frictional resistance coefficient (dimensionless). Returns 0 if Re < 1000.
    """
    re = reynolds_number(speed_m_s, temperature_c, length_m, salinity_g_kg)
    if re < 1000:
        return 0.0

    log_re = math.log10(re)
    cf = 0.075 / (log_re - 2.0) ** 2
    return cf


def compute_water_physics_features(speed_kn: float, sst_c: float,
                                   salinity_g_kg: float = SALINITY_PASIG_RIVER) -> dict[str, float]:
    """Compute all water physics features for a single observation.

    Args:
        speed_kn: Vessel speed in knots.
        sst_c: Sea surface temperature in degC.
        salinity_g_kg: Absolute salinity in g/kg.

    Returns:
        Dict with keys: water_density, kinematic_viscosity, log_reynolds_number,
        friction_coefficient.
    """
    speed_ms = speed_kn * 0.514444  # knots to m/s

    rho = seawater_density(sst_c, salinity_g_kg)
    nu = kinematic_viscosity(sst_c, salinity_g_kg)
    re = reynolds_number(speed_ms, sst_c, salinity_g_kg=salinity_g_kg)
    cf = friction_coefficient(speed_ms, sst_c, salinity_g_kg=salinity_g_kg)

    return {
        "water_density": rho,
        "kinematic_viscosity": nu * 1e6,  # scale to 1e-6 m²/s for numerical stability
        "log_reynolds_number": math.log10(re) if re > 0 else 0.0,
        "friction_coefficient": cf * 1e3,  # scale to 1e-3 for numerical stability
    }


if __name__ == "__main__":
    # Quick validation against ITTC standard values
    print("=== ITTC Validation ===")

    # Fresh water at 20degC (ITTC standard)
    rho_fw = seawater_density(20.0, 0.0)
    nu_fw = kinematic_viscosity(20.0, 0.0)
    print(f"Fresh water 20degC:  rho = {rho_fw:.3f} kg/m³ (expect ~998.2)")
    print(f"                   nu = {nu_fw:.6e} m²/s (expect ~1.003e-6)")

    # Seawater at 15degC, SA=35.16 g/kg (ITTC standard)
    rho_sw = seawater_density(15.0, 35.16)
    nu_sw = kinematic_viscosity(15.0, 35.16)
    print(f"\nSeawater 15degC/35ppt: rho = {rho_sw:.3f} kg/m³ (expect ~1025.9)")
    print(f"                   nu = {nu_sw:.6e} m²/s (expect ~1.189e-6)")

    # Manila Bay conditions (28degC, 10 g/kg)
    print(f"\n=== Pasig River Estimate (28degC, 10 g/kg) ===")
    feats = compute_water_physics_features(speed_kn=8.0, sst_c=28.0)
    for k, v in feats.items():
        print(f"  {k}: {v:.6f}")

    # Show sensitivity to SST (range we see in data)
    print(f"\n=== SST Sensitivity (10 g/kg, 8 kn) ===")
    for sst in [26, 27, 28, 29, 30, 31, 32]:
        f = compute_water_physics_features(speed_kn=8.0, sst_c=float(sst))
        print(f"  SST={sst}degC: rho={f['water_density']:.2f}, nu={f['kinematic_viscosity']:.4f}e-6, CF={f['friction_coefficient']:.4f}e-3")
