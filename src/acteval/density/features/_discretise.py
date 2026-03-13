"""Pure-numpy discretisation helpers (no torch dependency)."""

from typing import Iterable

import numpy as np
import pandas as pd


def descretise_population(
    data: pd.DataFrame, duration: int, step_size: int, class_map: dict
) -> np.ndarray:
    """Convert given population of activity traces into array [P, C, H, W].

    P is the population size.
    C (channel) is length 1.
    H is time steps.
    W is a one-hot encoding of activity type.

    Args:
        data (pd.DataFrame): Schedule DataFrame with pid, act, start, end columns.
        duration (int): Total duration in time units.
        step_size (int): Step size for discretisation.
        class_map (dict): Mapping from activity string to integer class index.

    Returns:
        np.ndarray: Array of shape [P, C, H, W].
    """
    persons = data.pid.nunique()
    num_classes = len(class_map)
    steps = duration // step_size
    encoded = np.zeros((persons, steps, num_classes, 1), dtype=np.float32)

    for pid, (_, trace) in enumerate(data.groupby("pid")):
        trace_encoding = descretise_trace(
            acts=trace.act,
            starts=trace.start,
            ends=trace.end,
            length=duration,
            class_map=class_map,
        )
        trace_encoding = down_sample(trace_encoding, step_size)
        trace_encoding = one_hot(trace_encoding, num_classes)
        trace_encoding = trace_encoding.reshape(steps, num_classes, 1)
        encoded[pid] = trace_encoding  # [B, H, W, C]
    encoded = encoded.transpose(0, 3, 1, 2)  # [B, C, H, W]
    return encoded


def descretise_trace(
    acts: Iterable[str],
    starts: Iterable[int],
    ends: Iterable[int],
    length: int,
    class_map: dict,
) -> np.ndarray:
    """Create categorical encoding from ranges with step of 1.

    Args:
        acts (Iterable[str]): Activity labels.
        starts (Iterable[int]): Start times.
        ends (Iterable[int]): End times.
        length (int): Total length of encoding.
        class_map (dict): Mapping from activity string to integer class index.

    Returns:
        np.ndarray: 1D categorical encoding of length `length`.
    """
    encoding = np.zeros((length), dtype=np.int8)
    for act, start, end in zip(acts, starts, ends):
        encoding[start:end] = class_map[act]
    return encoding


def down_sample(array: np.ndarray, step: int) -> np.ndarray:
    """Down-sample by stepping through given array.

    Args:
        array (np.ndarray): Input array.
        step (int): Step size.

    Returns:
        np.ndarray: Down-sampled array.
    """
    return array[::step]


def one_hot(target: np.ndarray, num_classes: int) -> np.ndarray:
    """One hot encoding of given categorical array.

    Args:
        target (np.ndarray): Categorical array.
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: One-hot encoded array.
    """
    return np.eye(num_classes)[target]
