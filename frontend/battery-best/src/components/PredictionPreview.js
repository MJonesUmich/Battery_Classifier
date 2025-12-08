import { Paper, Stack, Typography } from '@mui/material';

const PredictionPreview = ({ prediction, topProbability, probabilityEntries }) => (
  <Paper
    variant="outlined"
    sx={{
      p: { xs: 3, md: 4 },
      borderStyle: 'dashed',
      borderColor: 'primary.light',
      backgroundColor: 'rgba(25, 118, 210, 0.04)',
    }}
  >
    <Typography variant="subtitle1" fontWeight={600} gutterBottom>
      Prediction Preview
    </Typography>
    {prediction ? (
      <Stack spacing={2}>
        <Stack spacing={0.5}>
          <Typography variant="h4" fontWeight={700}>
            {prediction}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Top prediction with probability {topProbability ? topProbability.toFixed(2) : '--'}%
          </Typography>
        </Stack>
        <Stack direction="row" spacing={4} alignItems="center" sx={{ maxWidth: 400 }}>
          {probabilityEntries.map((entry) => (
            <Stack key={entry.label} spacing={0.5} alignItems="flex-start">
              <Typography variant="caption" color="text.secondary" fontWeight={600}>
                {entry.label}
              </Typography>
              <Typography variant="body2" fontWeight={600}>
                {entry.percent}%
              </Typography>
            </Stack>
          ))}
        </Stack>
      </Stack>
    ) : (
      <Typography variant="body2" color="text.secondary">
        Upload both charge and discharge CSV files from the same cycle to preview the chemistry classification
        probabilities.
      </Typography>
    )}
  </Paper>
);

export default PredictionPreview;

