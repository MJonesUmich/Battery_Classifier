import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import { Paper, Stack, Typography } from '@mui/material';

const HowItWorksCard = () => (
  <Paper elevation={3} sx={{ p: { xs: 3, md: 4 } }}>
    <Stack spacing={2.5}>
      <Stack direction="row" spacing={1.5} alignItems="flex-start">
        <InfoOutlinedIcon color="primary" fontSize="large" />
        <Stack spacing={0.75}>
          <Typography variant="h5" fontWeight={600}>
            How It Works
          </Typography>
          <Typography variant="body2" color="text.secondary">
            1) Provide charge and discharge CSVs with voltage, C-rate, and temperature, or a single-row CSV containing
            the 11 engineered features used by the model.
          </Typography>
          <Typography variant="body2" color="text.secondary">
            2) Use &quot;Predict Now&quot; on a sample to auto-run, or upload your own files and click &quot;Run Model
            Analysis&quot; to start inference.
          </Typography>
          <Typography variant="body2" color="text.secondary">
            3) View the predicted chemistry and class probabilities. Charts are omitted in this demo to keep the flow
            focused on the classifier.
          </Typography>
        </Stack>
      </Stack>
    </Stack>
  </Paper>
);

export default HowItWorksCard;

