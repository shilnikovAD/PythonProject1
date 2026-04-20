"""Физические проверки решения MoM для задачи двух сфер.

Запуск:
    python verify_physics.py --n1 250 --n2 250 --r1 0.05 --r2 0.05 --d 0.2 --dv 100 --ngrid 120

Что проверяем:
1) q1 + q2 ~ 0 (заряды электродов равны по модулю и противоположны)
2) Потенциал на поверхности каждой сферы близок к заданному (среднее/СКО/макс. отклонение)
3) Гаусс: поток E через контрольную сферу вокруг каждого электрода ~= Q/eps0
4) В вакууме (вне электродов) div(E) ~ 0 (приближенно, на сетке)
5) Вке: E примерно направлено от + к - (по знаку проекции на ось между центрами)

Скрипт не использует web-часть и позволяет понять, «физичная» ли картинка.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np

import main


@dataclass
class SurfaceStats:
    mean: float
    std: float
    max_abs: float


def surface_potential_stats(points_eval: np.ndarray, src_points: np.ndarray, q: np.ndarray, V_target: float) -> SurfaceStats:
    """Статистика ошибки потенциала на заданных точках поверхности.

    Важно: потенциал должен считаться от ВСЕХ зарядов системы (src_points, q),
    а не только от зарядов на той же сфере.
    """
    V = main.potential_at_points(points_eval, src_points, q)
    err = V - V_target
    return SurfaceStats(mean=float(np.mean(err)), std=float(np.std(err)), max_abs=float(np.max(np.abs(err))))


def gauss_flux_sphere(center: Tuple[float, float, float], radius: float, src_points: np.ndarray, q: np.ndarray, n_theta=40, n_phi=80) -> float:
    """Поток E через сферу радиуса radius (должен быть Q/eps0 для сферы, охватывающей заряд)."""
    cx, cy, cz = center
    thetas = np.linspace(1e-6, np.pi - 1e-6, n_theta)
    phis = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    tt, pp = np.meshgrid(thetas, phis, indexing="ij")

    x = cx + radius * np.sin(tt) * np.cos(pp)
    y = cy + radius * np.sin(tt) * np.sin(pp)
    z = cz + radius * np.cos(tt)

    pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    E = main.compute_field(pts, src_points, q)

    nvec = pts - np.array([cx, cy, cz])[None, :]
    nrm = np.linalg.norm(nvec, axis=1)
    nvec = nvec / nrm[:, None]

    # dS = r^2 sin(theta) dtheta dphi
    dtheta = thetas[1] - thetas[0]
    dphi = phis[1] - phis[0]
    dS = (radius ** 2) * np.sin(tt).ravel() * dtheta * dphi

    flux = float(np.sum(np.einsum("ij,ij->i", E, nvec) * dS))
    return flux


def divergence_on_grid(x: np.ndarray, y: np.ndarray, Ex: np.ndarray, Ey: np.ndarray) -> np.ndarray:
    """Приближенная дивергенция на 2D сетке (плоскость z=0), центральные разности внутри."""
    # Ex, Ey shape = (ny, nx)
    dEx_dx = np.gradient(Ex, x, axis=1)
    dEy_dy = np.gradient(Ey, y, axis=0)
    return dEx_dx + dEy_dy


def main_cli() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n1", type=int, default=250)
    p.add_argument("--n2", type=int, default=250)
    p.add_argument("--r1", type=float, default=0.05)
    p.add_argument("--r2", type=float, default=0.05)
    p.add_argument("--d", type=float, default=0.20)
    p.add_argument("--dv", type=float, default=100.0)
    p.add_argument("--ngrid", type=int, default=140)
    args = p.parse_args()

    res = main.solve_case(args.n1, args.n2, args.r1, args.r2, args.d, args.dv, args.ngrid)

    q1 = res["q1"]
    q2 = res["q2"]
    dv = args.dv

    c1 = (-0.5 * args.d, 0.0, 0.0)
    c2 = (+0.5 * args.d, 0.0, 0.0)
    v1 = +0.5 * dv
    v2 = -0.5 * dv

    # Пересобираем систему, чтобы получить src_points и q, без повторного solve_case
    el1 = main.sphere_panels(args.n1, args.r1, c1, v1)
    el2 = main.sphere_panels(args.n2, args.r2, c2, v2)
    A, b, src_points, split = main.build_mom_system(el1, el2)
    q = np.linalg.solve(A, b)

    print("=== Проверки физики (MoM, две сферы) ===")
    print(f"Q1 = {q1:.6e} Кл")
    print(f"Q2 = {q2:.6e} Кл")
    print(f"Q1+Q2 = {(q1+q2):.3e} Кл (должно быть ~0)")

    # 1) Потенциал на поверхности
    stats1 = surface_potential_stats(el1.points, src_points, q, v1)
    stats2 = surface_potential_stats(el2.points, src_points, q, v2)
    print("\n[Потенциал на поверхности] (ошибка: V_calc - V_target)")
    print(f"Сфера 1: mean={stats1.mean:+.3e} В, std={stats1.std:.3e} В, max|err|={stats1.max_abs:.3e} В")
    print(f"Сфера 2: mean={stats2.mean:+.3e} В, std={stats2.std:.3e} В, max|err|={stats2.max_abs:.3e} В")

    # 2) Гаусс
    r_gauss1 = args.r1 * 1.6
    r_gauss2 = args.r2 * 1.6
    flux1 = gauss_flux_sphere(c1, r_gauss1, src_points, q)
    flux2 = gauss_flux_sphere(c2, r_gauss2, src_points, q)

    Q1 = float(np.sum(q[:split]))
    Q2 = float(np.sum(q[split:]))

    print("\n[Закон Гаусса] поток через контрольную сферу")
    print(f"Flux1 = {flux1:.6e} В·м (Н·м^2/Кл),  Q1/eps0 = {Q1/main.EPS0:.6e}")
    print(f"Flux2 = {flux2:.6e} В·м (Н·м^2/Кл),  Q2/eps0 = {Q2/main.EPS0:.6e}")
    rel1 = abs(flux1 - Q1/main.EPS0) / (abs(Q1/main.EPS0) + 1e-30)
    rel2 = abs(flux2 - Q2/main.EPS0) / (abs(Q2/main.EPS0) + 1e-30)
    print(f"RelErr1 = {rel1:.3e}, RelErr2 = {rel2:.3e}")

    # 3) div(E) на сетке z=0, вне электродов
    lim = float(res["lim"])
    ngrid = int(res["grid"]["nx"])
    x = np.linspace(-lim, lim, ngrid)
    y = np.linspace(-lim, lim, ngrid)

    Ex = np.array(res["grid"]["ex"]).reshape(ngrid, ngrid)
    Ey = np.array(res["grid"]["ey"]).reshape(ngrid, ngrid)

    divE = divergence_on_grid(x, y, Ex, Ey)

    xx, yy = np.meshgrid(x, y)
    mask = (np.hypot(xx - c1[0], yy - c1[1]) > args.r1 * 1.05) & (np.hypot(xx - c2[0], yy - c2[1]) > args.r2 * 1.05)
    divE_masked = divE[mask]

    # Нормировка: |divE| / (|E|/L)
    Emag = np.hypot(Ex, Ey)
    Emag_m = Emag[mask]
    scale = (Emag_m / (lim + 1e-30))
    rel_div = np.abs(divE_masked) / (scale + 1e-30)

    print("\n[div(E) ~ 0 в вакууме] (2D оценка на z=0)")
    print(f"median(|divE|) = {np.median(np.abs(divE_masked)):.3e}")
    print(f"median(rel_div) = {np.median(rel_div):.3e}")
    print(f"95% quantile(rel_div) = {np.quantile(rel_div, 0.95):.3e}")

    # 4) Направление поля между центрами
    mid = np.array([0.0, 0.0, 0.0])
    E_mid = main.compute_field(mid[None, :], src_points, q)[0]
    print("\n[Направление поля в центре] E(0,0,0)")
    print(f"E = ({E_mid[0]:+.3e}, {E_mid[1]:+.3e}, {E_mid[2]:+.3e}) В/м")
    print("Ожидаем Ex > 0 (от + слева к - справа).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())

