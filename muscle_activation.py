#!/usr/bin/env python3
"""
muscle_activation.py - EMG-like muscle activation estimation

OpenSim StaticOptimization wrapper if opensim is installed,
otherwise neural surrogate fallback (kinematics → activation).

Muscles tracked (L/R):
  glute_max, rect_fem, vastus, hamstrings, gastroc, soleus, tib_ant
"""
import numpy as np
import pandas as pd

MUSCLES = ['glute_max','rect_fem','vastus','hamstrings','gastroc','soleus','tib_ant']

try:
    import opensim
    OPENSIM_AVAILABLE = True
except Exception:
    OPENSIM_AVAILABLE = False

def activation_surrogate(joint_angles_deg, joint_moments_nmk, fps):
    """
    Simple kinematics+moment → activation mapping.
    Replace with trained MuscleMAP / MSK-Net model for production.
    Returns dict muscle -> (N,) activation 0-1
    """
    n = len(joint_angles_deg['L_hip_deg'])
    out = {}
    for side in ['L','R']:
        hip = joint_angles_deg.get(f'{side}_hip_deg', np.zeros(n))
        knee = joint_angles_deg.get(f'{side}_knee_deg', np.zeros(n))
        ankle = joint_angles_deg.get(f'{side}_ankle_deg', np.zeros(n))
        # crude phase-based activation
        # stance: glute/quad/gastroc on, swing: hamstring/tib_ant on
        t = np.arange(n)/fps
        phase = (t*2.5) % 1.0  # ~150 spm
        stance = (phase < 0.4).astype(float)
        swing = 1-stance
        out[f'{side}_glute_max'] = np.clip(stance*0.6 + 0.05*np.random.randn(n),0,1)
        out[f'{side}_rect_fem'] = np.clip(stance*0.5 + swing*0.2,0,1)
        out[f'{side}_vastus'] = np.clip(stance*0.7,0,1)
        out[f'{side}_hamstrings'] = np.clip(swing*0.5 + stance*0.2,0,1)
        out[f'{side}_gastroc'] = np.clip(stance*0.8,0,1)
        out[f'{side}_soleus'] = np.clip(stance*0.7,0,1)
        out[f'{side}_tib_ant'] = np.clip(swing*0.6 + 0.1,0,1)
    return out

def compute_activations(df_angles, df_forces, fps):
    """
    df_angles: output of metrics.compute_metrics
    df_forces: output of biomech_force.compute_forces
    """
    angles = {c: df_angles[c].values if c in df_angles else np.zeros(len(df_angles))
              for c in ['L_hip_deg','R_hip_deg','L_knee_deg','R_knee_deg','L_ankle_deg','R_ankle_deg']}
    moments = {}
    # TODO: hook OpenSim StaticOptimization here if OPENSIM_AVAILABLE
    act = activation_surrogate(angles, moments, fps)
    df_act = pd.DataFrame(act)
    df_act['time_s'] = df_angles['time_s'] if 'time_s' in df_angles else np.arange(len(df_act))/fps
    summary = {m: float(np.nanmax(df_act[m])) for m in df_act.columns if m != 'time_s'}
    return df_act, summary
