// Corrected results chart - after fixing injection methodology
// Shows the corrected 0%/100% results with confident styling

const correctedData = [
  {
    trial: "Injection Trials",
    rate: 0.0,
    n: 48,
    color: "#5B8C5A"  // Forest green - confident
  },
  {
    trial: "Control Trials",
    rate: 100.0,
    n: 48,
    color: "#4A7C9B"  // Steel blue - confident
  }
];

function renderCorrectedResultsChart(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const width = Math.min(520, container.clientWidth || 520);
  const height = 220;
  const marginLeft = 130;
  const marginRight = 70;
  const marginTop = 50;
  const marginBottom = 40;
  const barHeight = 40;
  const barGap = 30;

  const chartWidth = width - marginLeft - marginRight;
  const scale = chartWidth / 100;

  let svg = `<svg width="${width}" height="${height}" style="font-family: system-ui, -apple-system, sans-serif;">`;

  // Title
  svg += `<text x="${width/2}" y="18" text-anchor="middle" fill="#333" font-size="14" font-weight="600">Corrected Results</text>`;

  // Subtitle with methodology note
  svg += `<text x="${width/2}" y="36" text-anchor="middle" fill="#666" font-size="12">After fixing injection methodology (N=48, preliminary)</text>`;

  // Grid lines
  for (let i = 0; i <= 100; i += 25) {
    const x = marginLeft + i * scale;
    svg += `<line x1="${x}" y1="${marginTop}" x2="${x}" y2="${height - marginBottom}" stroke="#e0e0e0" stroke-width="1"/>`;
    svg += `<text x="${x}" y="${height - marginBottom + 20}" text-anchor="middle" fill="#666" font-size="12">${i}%</text>`;
  }

  // X-axis label
  svg += `<text x="${marginLeft + chartWidth/2}" y="${height - 5}" text-anchor="middle" fill="#666" font-size="13">Success Rate</text>`;

  // Bars
  correctedData.forEach((d, i) => {
    const y = marginTop + i * (barHeight + barGap);
    const barWidth = Math.max(d.rate * scale, 2); // Minimum 2px for 0% bar visibility

    // Label
    svg += `<text x="${marginLeft - 10}" y="${y + barHeight/2 + 5}" text-anchor="end" fill="#333" font-size="14">${d.trial}</text>`;

    if (d.rate === 0) {
      // Special handling for 0% bar - show empty bar with border
      svg += `<g class="bar-group">
        <rect x="${marginLeft}" y="${y}" width="${chartWidth}" height="${barHeight}" fill="#f5f5f5" rx="4"/>
        <rect x="${marginLeft}" y="${y}" width="${chartWidth}" height="${barHeight}" fill="none" stroke="#ddd" stroke-width="1" rx="4"/>
        <text x="${marginLeft + 8}" y="${y + barHeight/2 + 5}" text-anchor="start" fill="${d.color}" font-size="14" font-weight="bold">0%</text>
        <text x="${marginLeft + chartWidth + 8}" y="${y + barHeight/2 + 5}" text-anchor="start" fill="#666" font-size="12">n=${d.n}</text>
      </g>`;

      // Add "No detection" annotation
      svg += `<text x="${marginLeft + chartWidth/2}" y="${y + barHeight/2 + 5}" text-anchor="middle" fill="#888" font-size="12" font-style="italic">No introspection detected</text>`;
    } else {
      // Solid bar with confident styling (no dashes)
      svg += `<g class="bar-group">
        <rect x="${marginLeft}" y="${y}" width="${barWidth}" height="${barHeight}" fill="${d.color}" rx="4"/>
        <text x="${marginLeft + barWidth - 8}" y="${y + barHeight/2 + 5}" text-anchor="end" fill="white" font-size="14" font-weight="bold">${d.rate.toFixed(0)}%</text>
        <text x="${marginLeft + barWidth + 8}" y="${y + barHeight/2 + 5}" text-anchor="start" fill="#666" font-size="12">n=${d.n}</text>
      </g>`;
    }
  });

  svg += '</svg>';
  container.innerHTML = svg;
}

// Auto-init
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => renderCorrectedResultsChart('corrected-results-chart'));
} else {
  renderCorrectedResultsChart('corrected-results-chart');
}
