import BusinessCenterIcon from '@mui/icons-material/BusinessCenter';
import SupportAgentIcon from '@mui/icons-material/SupportAgent';
import { Button, Chip, Grid, Link, Paper, Stack, Typography } from '@mui/material';

const EnterpriseCta = () => (
  <Paper elevation={0} sx={{ p: { xs: 3, md: 4 }, border: '1px solid', borderColor: 'divider' }}>
    <Stack spacing={3}>
      <Stack direction="row" spacing={1.5} alignItems="center">
        <BusinessCenterIcon color="primary" fontSize="large" />
        <Typography variant="h5" fontWeight={600}>
          Ready for Batch Update?
        </Typography>
      </Stack>
      <Typography variant="body2" color="text.secondary">
        Keep your production fleets synchronized with nightly batch updates, automated health reports, and seamless API
        integration. We tailor the deployment plan to your data cadence and infrastructure requirements.
      </Typography>
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 2.5, height: '100%' }}>
            <Stack spacing={1.5}>
              <Stack direction="row" spacing={1} alignItems="center">
                <Chip label="Batch Predict" color="primary" size="small" />
                <Typography variant="subtitle2" fontWeight={600}>
                  High-volume scoring
                </Typography>
              </Stack>
              <Typography variant="body2" color="text.secondary">
                Stream ingest-ready files and receive inference results through the batch update queue or via REST
                endpoints with secure credentials.
              </Typography>
            </Stack>
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 2.5, height: '100%' }}>
            <Stack spacing={1.5}>
              <Stack direction="row" spacing={1} alignItems="center">
                <Chip label="Enterprise Support" color="secondary" size="small" />
                <Typography variant="subtitle2" fontWeight={600}>
                  Co-design the rollout
                </Typography>
              </Stack>
              <Typography variant="body2" color="text.secondary">
                Collaborate with our engineering team to define monitoring SLAs, deployment guardrails, and integration
                checkpoints tailored to your roadmap.
              </Typography>
            </Stack>
          </Paper>
        </Grid>
      </Grid>
      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems="center">
        <Button
          variant="contained"
          startIcon={<SupportAgentIcon />}
          href="mailto:solutions@batteryinsight.ai?subject=Battery%20Insight%20Studio%20-%20Batch%20Predict"
        >
          Contact Solutions Team
        </Button>
        <Button variant="outlined" color="primary" href="" target="_blank" rel="noopener noreferrer">
          Schedule a Planning Call
        </Button>
        <Typography variant="caption" color="text.secondary">
          Prefer another channel? Reach us via{' '}
          <Link href="https://batteryinsight.ai" target="_blank" rel="noopener noreferrer">
            batteryinsight.ai
          </Link>
          .
        </Typography>
      </Stack>
    </Stack>
  </Paper>
);

export default EnterpriseCta;

