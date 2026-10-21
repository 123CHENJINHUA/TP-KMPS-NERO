"""Isaac-only check: save window positions, alter them, and restore the layout."""
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
try:
    import json
    import omni.ui as ui
    from sim_bridge.ui_layout import ROOT, save_layout, load_layout, build_layout_controls

    destination = ROOT / 'assets/layouts/.layout_verification.json'
    assert not destination.exists(), 'Do not overwrite an existing verification file'
    first = ui.Window('Layout Verification A', width=350, height=250,
                      position_x=50, position_y=80)
    second = ui.Window('Layout Verification B', width=300, height=200,
                       position_x=480, position_y=100)
    panel = ui.Window('Layout Verification Controls', width=350, height=150)
    with panel.frame:
        with ui.VStack():
            status = build_layout_controls()
    for _ in range(10):
        app.update()
    before = {window.title: (window.width, window.height, window.position_x, window.position_y)
              for window in (first, second)}
    save_layout(destination)
    assert isinstance(json.loads(destination.read_text()), list)
    first.width, first.height = 150, 100
    second.width, second.height = 160, 110
    extra = ui.Window('Layout Verification New Window', width=100, height=100)
    for _ in range(5):
        app.update()
    load_layout(destination)
    for _ in range(120):
        app.update()
    for window in (first, second):
        actual = (window.width, window.height, window.position_x, window.position_y)
        assert all(abs(a - b) <= 2 for a, b in zip(actual, before[window.title])), (window.title, before[window.title], actual)
        print(f'UI_LAYOUT_WINDOW_RESTORED title={window.title} bounds={actual}', flush=True)
    assert extra.visible, 'Loading a layout must keep new windows open'
    for window in (first, second, panel, extra):
        window.destroy()
    destination.unlink()
    print('UI_LAYOUT_INTEGRATION_PASS', flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    app.app.post_quit(1)
    raise
finally:
    app.close()
