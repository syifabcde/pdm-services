"""
The single PDM dashboard -- one page for every machine, one URL. Pick the machine and the
device at the top of the page; each machine's own sections come from machines/<name>/ (see
core/registry.py), so a new machine is a new folder there, not a new page.

Run from pdm-services/:
    python -m streamlit run dashboard.py
"""
import streamlit as st

from core import registry

st.set_page_config(page_title="PDM Monitor", layout="wide")

machines = registry.load_machines()
if not machines:
    st.error("Belum ada mesin terdaftar: buat machines/<nama>/machine.toml.")
    st.stop()

st.header("PDM Monitor")
machine_col, device_col, _ = st.columns([1, 1, 2])   # compact selectors, the rest of the row stays empty
key = machine_col.selectbox("Mesin", list(machines), format_func=lambda k: machines[k].label)
machine = machines[key]
iddev = device_col.selectbox("Device (iddev)", machine.devices)

render = registry.load_renderer(machine)

st.title(machine.label)
refresh = registry.refresh_seconds(machine, iddev)
st.caption(f"Device iddev={iddev} | auto refresh tiap 1 menit")

if hasattr(st, "fragment"):
    @st.fragment(run_every=refresh)
    def live_section():
        render(iddev)

    live_section()
else:
    render(iddev)
    st.button("Refresh manual")
