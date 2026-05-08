import logging
from typing import Optional

import supersuit as ss
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.utils import BaseWrapper
from visual_utils import VisualWrapper, set_distortion_level

logger = logging.getLogger("ml-project")


def create_environment(
    max_cycles: int = 2500,
    render_mode: Optional[str] = None,
    max_zombies: int = 4,
    frame_stack: Optional[int] = None,
    resize_dim: Optional[tuple[int, int]] = None,
    distortion_level: int = 0,
) -> BaseWrapper:
    """
    Create a configured KAZ environment.

    Args:
        num_agents: Number of archer agents (1 or 2)
        max_cycles: Maximum steps before episode truncation
        render_mode: None, "human", or "rgb_array"
        max_zombies: Maximum number of zombies in the arena
        visual_observation: Whether to use pixel observations
        frame_stack: Number of frames to stack (None for no stacking)
        resize_dim: Tuple (width, height) to resize visual observations

    Returns:
        A configured PettingZoo environment
    """
    # Set parameters

    num_agents = 2
    visual_observation = True


    # Create base environment
    env = knights_archers_zombies_v10.env(
        max_cycles=max_cycles,
        num_archers=num_agents,
        num_knights=0,
        max_zombies=max_zombies,
        vector_state=not visual_observation,
        render_mode= "rgb_array" # We will handle rendering in VisualWrapper, not here
    )

    # Apply visual observation wrapper
    set_distortion_level(level=distortion_level)
    env = VisualWrapper(env)
    env.render_mode = render_mode

    # Handle agent termination
    env = ss.black_death_v3(env)




    logger.info(
        f"Created KAZ environment with {num_agents} agents and max {max_zombies} zombies. "
        f"Observation type: {'visual' if visual_observation else 'vector'}"
    )

    return env


def iou(r1, r2):
    l1, t1, w1, h1 = r1
    r1 = l1 + w1
    b1 = t1 + h1

    l2, t2, w2, h2 = r2
    r2 = l2 + w2
    b2 = t2 + h2

    xl = max(l1, l2)
    yt = max(t1, t2)
    xr = min(r1, r2)
    yb = min(b1, b2)

    if xr < xl or yb < yt:
        return 0.0

    intersect_area = (xr - xl) * (yb - yt)
    bb1a = (r1 - l1) * (b1 - t1)
    bb2a = (r2 - l2) * (b2 - t2)
    iou = intersect_area / float(bb1a + bb2a - intersect_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

