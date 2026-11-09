"""Small A/A sanity check for the released muse_decode_layer entry."""

SPEC = dict(
    name="muse_layer_aa",
    problem="muse_decode_layer",
    entry="submissions/muse_decode_layer.py",
    recipe="layer",
    arms=[],
    shapes=["d1s"],
    passes=[dict(tag="aa", nsets=3)],
    rounds=24,
    nearby=False,
    raw=False,
    predeclared=dict(
        shape="d1s",
        arm="twin",
        metric="ratio",
        op="<",
        value=0.97,
        text="A/A twin ratio < 0.97 at d1s indicates an out-of-band measurement",
    ),
)
