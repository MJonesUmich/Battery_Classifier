import { Box, Grid, Paper, Stack, Typography } from '@mui/material';
import VoltagePlot from './VoltagePlot';

const PredictionPreview = ({ prediction, topProbability, probabilityEntries, chargePoints, dischargePoints }) => (
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
        <Typography variant="h4" fontWeight={700}>
          {prediction}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Top prediction with probability {topProbability ? topProbability.toFixed(2) : '--'}%
        </Typography>
        <Stack spacing={1}>
          {probabilityEntries.map((entry) => (
            <Stack key={entry.label} direction="row" spacing={1} alignItems="center" justifyContent="space-between">
              <Typography variant="body2" fontWeight={600}>
                {entry.label}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {entry.percent}%
              </Typography>
            </Stack>
          ))}
        </Stack>
        <Box>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <VoltagePlot title="Charge Voltage Profile" color="#1976d2" points={chargePoints} />
            </Grid>
            <Grid item xs={12} md={6}>
              <VoltagePlot title="Discharge Voltage Profile" color="#ff7043" points={dischargePoints} />
            </Grid>
          </Grid>
        </Box>
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

