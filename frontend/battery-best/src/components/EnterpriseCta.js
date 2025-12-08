import BusinessCenterIcon from '@mui/icons-material/BusinessCenter';
import { Button, Chip, Paper, Stack, Typography } from '@mui/material';

const EnterpriseCta = () => (
  <Paper elevation={0} sx={{ p: { xs: 3, md: 4 }, border: '1px solid', borderColor: 'divider' }}>
    <Stack spacing={3}>
      <Stack direction="row" spacing={1.5} alignItems="center">
        <BusinessCenterIcon color="primary" fontSize="large" />
        <Typography variant="h5" fontWeight={600}>
          Project Notes
        </Typography>
      </Stack>
      <Typography variant="body2" color="text.secondary">
        This front-end showcases a chemistry classification prototype. Batch pipelines, APIs, and support flows are out
        of scope for this demo.
      </Typography>
      <Paper variant="outlined" sx={{ p: 2.5 }}>
        <Stack spacing={1.5}>
          <Stack direction="row" spacing={1} alignItems="center">
            <Chip label="Future work" color="primary" size="small" />
            <Typography variant="subtitle2" fontWeight={600}>
              More features on the way
            </Typography>
          </Stack>
          <Typography variant="body2" color="text.secondary">
            Additional visualizations, batch processing, and extended models are in development. Stay tuned.
          </Typography>
          <Button
            variant="contained"
            color="primary"
            href="https://github.com/MJonesUmich/Battery_Classifier"
            target="_blank"
            rel="noopener noreferrer"
          >
            View project repo
          </Button>
        </Stack>
      </Paper>
    </Stack>
  </Paper>
);

export default EnterpriseCta;

