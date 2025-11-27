import { Box, Typography } from '@mui/material';

const VoltagePlot = ({ title, color, points = [], axisColor = '#9e9e9e' }) => {
  const width = 350;
  const height = 220;
  const padding = 40;
  const tickCount = 4;

  const processedPoints = points
    .map((point, idx) => ({
      x: Number.isFinite(point.x) ? Number(point.x) : Number(point.sample_index ?? idx),
      y: Number(point.y),
    }))
    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));

  if (!processedPoints.length) {
    return (
      <Box>
        <Typography variant="subtitle2" fontWeight={600} gutterBottom>
          {title}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Not enough data to render this phase.
        </Typography>
      </Box>
    );
  }

  const xValues = processedPoints.map((point) => point.x);
  const yValues = processedPoints.map((point) => point.y);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  const coords = processedPoints.map((point) => {
    const svgX = padding + ((point.x - xMin) / xRange) * (width - padding * 2);
    const svgY = height - padding - ((point.y - yMin) / yRange) * (height - padding * 2);
    return { svgX, svgY, rawX: point.x, rawY: point.y };
  });

  const ticks = (min, max) =>
    Array.from({ length: tickCount }, (_, idx) => min + (idx / (tickCount - 1)) * (max - min));
  const xTicks = ticks(xMin, xMax);
  const yTicks = ticks(yMin, yMax);

  return (
    <Box>
      <Typography variant="subtitle2" fontWeight={600} gutterBottom>
        {title}
      </Typography>
      <Box sx={{ width: '100%', overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height}>
          <rect
            x={padding}
            y={padding / 2}
            width={width - padding * 2}
            height={height - padding * 1.5}
            fill="#fafafa"
            stroke="#e0e0e0"
            rx={6}
          />

          {xTicks.map((tick) => {
            const xPos = padding + ((tick - xMin) / xRange) * (width - padding * 2);
            return (
              <line
                key={`grid-x-${tick}`}
                x1={xPos}
                x2={xPos}
                y1={padding / 2}
                y2={height - padding}
                stroke="#eeeeee"
                strokeDasharray="4 4"
              />
            );
          })}
          {yTicks.map((tick) => {
            const yPos = height - padding - ((tick - yMin) / yRange) * (height - padding * 2);
            return (
              <line
                key={`grid-y-${tick}`}
                x1={padding}
                x2={width - padding}
                y1={yPos}
                y2={yPos}
                stroke="#eeeeee"
                strokeDasharray="4 4"
              />
            );
          })}

          <polyline
            points={coords.map((coord) => `${coord.svgX},${coord.svgY}`).join(' ')}
            fill="none"
            stroke={color}
            strokeWidth={2.5}
            strokeLinejoin="round"
            strokeLinecap="round"
          />

          {xTicks.map((tick) => {
            const xPos = padding + ((tick - xMin) / xRange) * (width - padding * 2);
            return (
              <text
                key={`x-${tick}`}
                x={xPos}
                y={height - padding + 16}
                textAnchor="middle"
                fill={axisColor}
                fontSize="11"
              >
                {tick.toFixed(2)}
              </text>
            );
          })}

          {yTicks.map((tick) => {
            const yPos = height - padding - ((tick - yMin) / yRange) * (height - padding * 2);
            return (
              <text
                key={`y-${tick}`}
                x={padding - 8}
                y={yPos + 4}
                textAnchor="end"
                fill={axisColor}
                fontSize="11"
              >
                {tick.toFixed(2)}
              </text>
            );
          })}

          <text
            x={width / 2}
            y={height - 4}
            textAnchor="middle"
            fill={axisColor}
            fontSize="11"
            fontWeight={600}
          >
            Normalized Time / Sample Index
          </text>
          <text
            x={12}
            y={padding / 2 - 6}
            textAnchor="start"
            fill={axisColor}
            fontSize="11"
            fontWeight={600}
          >
            Voltage (V)
          </text>
        </svg>
      </Box>
    </Box>
  );
};

export default VoltagePlot;

