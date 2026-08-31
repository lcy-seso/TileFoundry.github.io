from __future__ import annotations

from tilefoundry import func, module
from tilefoundry.dsl import Tensor, tf
from tilefoundry.ir.types.shard import Topology
from tilefoundry.target import CudaTarget

T, D, CTAS = 1024, 256, 8


@module(entry="chain", target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", CTAS),))
class Unplaced:
    @func
    def chain(x: Tensor[(1, T, D), "f16"]) -> Tensor[(1, T, D), "f16"]:
        a = tf.mul(x, x)
        b = tf.add(a, x)
        return tf.mul(b, x)
