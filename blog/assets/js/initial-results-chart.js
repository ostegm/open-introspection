// Initial results chart - with skeptical framing
// Shows the "too good to be true" 50%/94% results that turned out to be methodologically flawed
// Anthropic's paper showed ~20% for much larger models - these results were suspiciously high

const initialData = [
  {
    trial: "Injection Trials",
    rate: 50.0,
    n: 96,
    color: "#B8956E",     // Muted dusty orange - skeptical tone
    borderColor: "#8B7355" // Darker border for dashed effect
  },
  {
    trial: "Control Trials",
    rate: 94.0,
    n: 96,
    color: "#7A9EA3",     // Muted sage teal
    borderColor: "#5A7E83"
  }
];

function renderInitialResultsChart(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const width = Math.min(520, container.clientWidth || 520);
  const height = 220;
  const marginLeft = 130;
  const marginRight = 80;
  const marginTop = 45;
  const marginBottom = 40;
  const barHeight = 40;
  const barGap = 30;

  const chartWidth = width - marginLeft - marginRight;
  const scale = chartWidth / 100;

  let svg = `<svg width="${width}" height="${height}" style="font-family: system-ui, -apple-system, sans-serif;">`;

  // Title with skeptical framing
  svg += `<text x="${width/2}" y="16" text-anchor="middle" fill="#555" font-size="14" font-weight="500">Initial Results</text>`;
  svg += `<text x="${width/2}" y="34" text-anchor="middle" fill="#888" font-size="12" font-style="italic">(something seems off...)</text>`;

  // Grid lines
  for (let i = 0; i <= 100; i += 25) {
    const x = marginLeft + i * scale;
    svg += `<line x1="${x}" y1="${marginTop}" x2="${x}" y2="${height - marginBottom}" stroke="#e5e5e5" stroke-width="1"/>`;
    svg += `<text x="${x}" y="${height - marginBottom + 20}" text-anchor="middle" fill="#888" font-size="11">${i}%</text>`;
  }

  // X-axis label
  svg += `<text x="${marginLeft + chartWidth/2}" y="${height - 5}" text-anchor="middle" fill="#666" font-size="12">Success Rate</text>`;

  // Bars
  initialData.forEach((d, i) => {
    const y = marginTop + i * (barHeight + barGap);
    const barWidth = d.rate * scale;

    // Label
    svg += `<text x="${marginLeft - 10}" y="${y + barHeight/2 + 5}" text-anchor="end" fill="#444" font-size="13">${d.trial}</text>`;

    // Bar with reduced opacity and dashed border to indicate uncertainty
    svg += `<g class="bar-group">
      <rect x="${marginLeft}" y="${y}" width="${barWidth}" height="${barHeight}" fill="${d.color}" rx="4" opacity="0.7"/>
      <rect x="${marginLeft}" y="${y}" width="${barWidth}" height="${barHeight}" fill="none" stroke="${d.borderColor}" stroke-width="2" stroke-dasharray="6,3" rx="4"/>
      <text x="${marginLeft + barWidth - 8}" y="${y + barHeight/2 + 5}" text-anchor="end" fill="white" font-size="14" font-weight="bold">${d.rate.toFixed(0)}%</text>
      <text x="${marginLeft + barWidth + 8}" y="${y + barHeight/2 + 5}" text-anchor="start" fill="#777" font-size="11" font-style="italic">n=${d.n}</text>
    </g>`;
  });

  // Question mark annotation with circle background - positioned to the right of injection bar
  const questionX = marginLeft + 50 * scale + 50;
  const questionY = marginTop + barHeight/2;
  svg += `<circle cx="${questionX}" cy="${questionY}" r="12" fill="#F5E6D3" stroke="#C9A86C" stroke-width="1.5"/>`;
  svg += `<text x="${questionX}" y="${questionY + 5}" text-anchor="middle" fill="#A67C3D" font-size="16" font-weight="bold">?</text>`;

  // Subtle annotation line connecting question mark to bar
  svg += `<line x1="${marginLeft + 50 * scale + 8}" y1="${questionY}" x2="${questionX - 14}" y2="${questionY}" stroke="#C9A86C" stroke-width="1" stroke-dasharray="3,2" opacity="0.7"/>`;

  svg += '</svg>';
  container.innerHTML = svg;
}

// Auto-init
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => renderInitialResultsChart('initial-results-chart'));
} else {
  renderInitialResultsChart('initial-results-chart');
}
