import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
from scipy.linalg import solve

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

EPS0 = 8.8541878128e-12
K_COULOMB = 1.0 / (4.0 * np.pi * EPS0)


@dataclass
class ElectrodePanels:
    points: np.ndarray  # (N, 3)
    area: np.ndarray    # (N,)
    potential: float


class SimRequest(BaseModel):
    n1: int = Field(120, ge=20, le=800)
    n2: int = Field(120, ge=20, le=800)
    r1: float = Field(0.05, gt=0.0)
    r2: float = Field(0.05, gt=0.0)
    d: float = Field(0.20, gt=0.0)
    dv: float = Field(100.0)
    ngrid: int = Field(90, ge=40, le=260)


def fibonacci_sphere(n: int, radius: float, center: Tuple[float, float, float]) -> np.ndarray:
    i = np.arange(n)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, None))
    theta = phi * i
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    pts = np.column_stack((x, y, z)) * radius
    return pts + np.array(center)


def sphere_panels(n: int, radius: float, center: Tuple[float, float, float], potential: float) -> ElectrodePanels:
    points = fibonacci_sphere(n, radius, center)
    area = np.full(n, 4.0 * np.pi * radius * radius / n)
    return ElectrodePanels(points=points, area=area, potential=potential)


def build_mom_system(el1: ElectrodePanels, el2: ElectrodePanels, self_radius_factor: float = 0.35):
    points = np.vstack([el1.points, el2.points])
    areas = np.concatenate([el1.area, el2.area])
    n1 = el1.points.shape[0]
    n = points.shape[0]

    b = np.concatenate([
        np.full(n1, el1.potential),
        np.full(n - n1, el2.potential),
    ])

    diff = points[:, None, :] - points[None, :, :]
    r = np.linalg.norm(diff, axis=2)

    # Self-term regularization for collocation approximation.
    a_eff = np.sqrt(areas / np.pi) * self_radius_factor
    np.fill_diagonal(r, a_eff)

    A = K_COULOMB / r
    return A, b, points, n1


def potential_at_points(points_eval: np.ndarray, src_points: np.ndarray, q: np.ndarray) -> np.ndarray:
    diff = points_eval[:, None, :] - src_points[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    r = np.where(r > 1e-16, r, np.inf)
    return K_COULOMB * np.sum(q[None, :] / r, axis=1)


def compute_field(points_eval: np.ndarray, src_points: np.ndarray, q: np.ndarray) -> np.ndarray:
    diff = points_eval[:, None, :] - src_points[None, :, :]
    r2 = np.sum(diff * diff, axis=2)
    r = np.sqrt(r2)
    r3 = np.where(r > 1e-16, r2 * r, np.inf)
    e = K_COULOMB * np.sum((q[None, :, None] * diff) / r3[:, :, None], axis=1)
    return e


def solve_case(n1: int, n2: int, r1: float, r2: float, d: float, dv: float, ngrid: int) -> Dict[str, Any]:
    if d <= r1 + r2:
        raise ValueError("Сферы пересекаются: требуется d > r1 + r2.")
    if abs(dv) < 1e-15:
        raise ValueError("Разность потенциалов dv должна быть отлична от нуля.")

    c1 = (-0.5 * d, 0.0, 0.0)
    c2 = (+0.5 * d, 0.0, 0.0)
    v1 = +0.5 * dv
    v2 = -0.5 * dv

    el1 = sphere_panels(n1, r1, c1, v1)
    el2 = sphere_panels(n2, r2, c2, v2)

    A, b, points, split = build_mom_system(el1, el2)
    q = solve(A, b, assume_a="gen")

    q1 = float(np.sum(q[:split]))
    q2 = float(np.sum(q[split:]))
    capacitance = float(abs(q1) / abs(dv))

    lim = max(1.2 * (0.5 * d + max(r1, r2)), 2.5 * max(r1, r2))

    x = np.linspace(-lim, lim, ngrid)
    y = np.linspace(-lim, lim, ngrid)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    peval = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    v = potential_at_points(peval, points, q)
    e = compute_field(peval, points, q)

    ex = e[:, 0]
    ey = e[:, 1]

    return {
        "q1": q1,
        "q2": q2,
        "capacitance": capacitance,
        "lim": lim,
        "geometry": {
            "c1": [c1[0], c1[1]],
            "c2": [c2[0], c2[1]],
            "r1": r1,
            "r2": r2,
        },
        "grid": {
            "nx": int(ngrid),
            "ny": int(ngrid),
            "potential": v.tolist(),
            "ex": ex.tolist(),
            "ey": ey.tolist(),
        },
    }


app = FastAPI(title="Electrostatics MoM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/")
def root():
    return FileResponse("index.html")


@app.post("/api/simulate")
def api_simulate(req: SimRequest):
    try:
        return solve_case(
            n1=req.n1,
            n2=req.n2,
            r1=req.r1,
            r2=req.r2,
            d=req.d,
            dv=req.dv,
            ngrid=req.ngrid,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка расчета: {exc}")


def run_cli():
    parser = argparse.ArgumentParser(description="Электростатика MoM: две сферы")
    parser.add_argument("--n1", type=int, default=250)
    parser.add_argument("--n2", type=int, default=250)
    parser.add_argument("--r1", type=float, default=0.05)
    parser.add_argument("--r2", type=float, default=0.05)
    parser.add_argument("--d", type=float, default=0.20)
    parser.add_argument("--dv", type=float, default=100.0)
    parser.add_argument("--ngrid", type=int, default=120)
    args = parser.parse_args()

    result = solve_case(args.n1, args.n2, args.r1, args.r2, args.d, args.dv, args.ngrid)
    print("=== CLI расчет MoM ===")
    print(f"Q1 = {result['q1']:.6e} Кл")
    print(f"Q2 = {result['q2']:.6e} Кл")
    print(f"Q1+Q2 = {(result['q1'] + result['q2']):.3e} Кл")
    print(f"C = {result['capacitance']:.6e} Ф")


if __name__ == "__main__":
    run_cli()
