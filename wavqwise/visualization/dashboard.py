"""Auto-generate HTML dashboard."""
class DashboardBuilder:
    def build(self, data, target, time_col, output="dashboard.html", **kwargs):
        html = f"<html><head><title>WavqWise Dashboard</title></head><body>"
        html += f"<h1>WavqWise Dashboard</h1>"
        html += f"<p>Target: {target} | Rows: {len(data)}</p>"
        html += f"</body></html>"
        with open(output, "w") as f:
            f.write(html)
