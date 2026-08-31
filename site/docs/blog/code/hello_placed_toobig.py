from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Mesh, Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

T, D, CTAS = 4096, 2048, 8


@module(entry="chain", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", CTAS),))
class Placed:
    @func
    def chain(x: Tensor[(1, T, D), "f16"]) -> Tensor[(1, T, D), "f16"]:
        with Mesh(("cta",), layout=(CTAS,), names=("tile",)) as cta:
            xr = tf.reshard(x, (1, T @ cta.tile, D), "rmem")
            a = tf.mul(xr, xr)
            b = tf.add(a, xr)
            return tf.reshard(tf.mul(b, xr), (1, T @ cta.tile, D), "gmem")
