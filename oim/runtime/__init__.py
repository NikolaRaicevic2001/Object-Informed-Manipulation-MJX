"""What every world needs to *run* something, independent of which world.

Nothing here knows about push-T, about ADMM, or about which dynamics a run
uses -- which is the point. If two worlds record, render, log or size a
sampler differently, a difference between their results is not a difference
in dynamics, and the comparison the repo exists to make stops meaning
anything.

    logs.py       the MuJoCo run log's schema and its three writers
    mjcf.py       camera/mocap/keyframe lookups, and the execution model
    overlay.py    plan trajectories drawn into a MuJoCo scene
    samplers.py   build_sub_optimizer, object_sample_count, consensus_space
    video.py      the ffmpeg pipe, and the offscreen renderer that feeds it
    viewer.py     the interactive closed loop (`run_interactive`)
    viewer_async.py   the same, planner and simulator in separate processes

Deliberately importable without choosing a world: `oim.worlds.*` depends on
`oim.runtime`, never the reverse.
"""
