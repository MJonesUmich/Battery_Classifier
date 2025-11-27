import { Box, Chip, Typography } from '@mui/material';

const HeroBanner = ({ logoSrc }) => (
  <Box textAlign="center">
    <Box
      component="img"
      src={logoSrc}
      alt="Battery Insight Studio logo"
      sx={{
        width: { xs: 72, sm: 84, md: 600 },
        height: 'auto',
        mx: 'auto',
        mb: 2,
      }}
    />

    <Typography variant="body1" color="text.secondary">
      Upload raw operational or lab data from your batteries. The platform will clean the signals,
      train diagnostic models, and estimate the current health to help you evaluate fleet performance in minutes.
    </Typography>
    <Chip label="Trial Workspace" color="primary" variant="outlined" sx={{ mt: 2, fontWeight: 600 }} />
  </Box>
);

export default HeroBanner;

