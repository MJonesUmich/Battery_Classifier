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
            1. Prepare a dataset containing fields such as time, voltage, current, and temperature in CSV or XLSX format.
          </Typography>
          <Typography variant="body2" color="text.secondary">
            2. Click the upload button below, verify the preview, and start the analysis.
          </Typography>
          <Typography variant="body2" color="text.secondary">
            3. The model will return health index, remaining life, and other KPIs (placeholder UI in this version).
          </Typography>
        </Stack>
      </Stack>
    </Stack>
  </Paper>
);

export default HowItWorksCard;

