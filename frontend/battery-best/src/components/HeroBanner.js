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
      Demo: classify battery chemistry from charge/discharge records. Upload your CSVs or run the provided samples to see
      the logistic regression output based on 11 engineered features.
    </Typography>
    <Chip label="Chemistry Classifier Demo" color="primary" variant="outlined" sx={{ mt: 2, fontWeight: 600 }} />
  </Box>
);

export default HeroBanner;

