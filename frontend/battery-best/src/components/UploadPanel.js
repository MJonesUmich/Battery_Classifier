import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { Alert, Button, Chip, CircularProgress, Divider, Grid, Paper, Stack, Typography } from '@mui/material';

const UploadPanel = ({
  fileInputRef,
  selectedFileName,
  isProcessing,
  sampleDatasets,
  datasetsLoading,
  datasetsError,
  phaseSummaries,
  errorMessage,
  statusMessage,
  onSelectFile,
  onFileChange,
  onSamplePredict,
  onRunModel,
  onDismissError,
  onDismissStatus,
}) => (
  <Paper elevation={1} sx={{ p: { xs: 3, md: 4 } }}>
    <Stack spacing={3}>
      <Stack spacing={1}>
        <Typography variant="h5" fontWeight={600}>
          Upload Dataset
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Supports CSV files. Parsing and inference run in-browser for this demo; upload charge and discharge together, or
          a single-row feature CSV.
        </Typography>
        <Typography variant="caption" color="text.secondary">
          Demo mode processes one sample at a time; files are not sent to any server.
        </Typography>
      </Stack>

      <Paper
        variant="outlined"
        sx={{
          p: { xs: 2, sm: 3 },
          backgroundColor: 'grey.50',
          borderStyle: 'dashed',
        }}
      >
        <Stack spacing={2}>
          <Typography variant="subtitle2" fontWeight={600}>
            Need a dataset to try?
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Grab one of the sample CSV files below if you don't have your own data yet, then come back to upload.
          </Typography>
          {datasetsLoading ? (
            <Stack direction="row" spacing={1} alignItems="center">
              <CircularProgress size={20} />
              <Typography variant="body2" color="text.secondary">
                Loading sample datasets...
              </Typography>
            </Stack>
          ) : datasetsError ? (
            <Alert severity="warning">{datasetsError}</Alert>
          ) : sampleDatasets.length ? (
            <Grid container spacing={2}>
              {sampleDatasets.map((dataset) => (
                <Grid item xs={6} key={dataset.id}>
                  <Paper
                    variant="outlined"
                    sx={{
                      p: 2,
                      height: 175,
                      width: 200,
                      display: 'flex',
                      flexDirection: 'column',
                    }}
                  >
                    <Stack spacing={1.5} sx={{ flexGrow: 1, overflow: 'hidden' }}>
                      <Stack
                        direction="row"
                        spacing={1}
                        alignItems="center"
                        justifyContent="space-between"
                        sx={{ width: '100%', overflow: 'hidden' }} // Ensure container clips overflow
                      >
                        <Typography
                          variant="subtitle2"
                          fontWeight={600}
                          sx={{
                            flex: 1,
                            minWidth: 0, // Critical for flex child to shrink
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                          title={dataset.title}
                        >
                          {dataset.title}
                        </Typography>
                        <Chip
                          label={dataset.size || 'CSV'}
                          size="small"
                          color="primary"
                          variant="outlined"
                          sx={{ flexShrink: 0 }}
                        />
                      </Stack>
                      <Typography variant="body2" color="text.secondary" sx={{ overflow: 'hidden' }}>
                        {dataset.description}
                      </Typography>
                    </Stack>
                    <Stack direction="row" spacing={1} sx={{ mt: 2 }}>
                      <Button component="a" href={dataset.url} download size="small" variant="outlined" color="primary" fullWidth>
                        Download
                      </Button>
                      <Button
                        size="small"
                        variant="contained"
                        color="primary"
                        fullWidth
                        onClick={() => onSamplePredict(dataset)}
                        disabled={isProcessing}
                      >
                        Predict Now
                      </Button>
                    </Stack>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          ) : (
            <Typography variant="body2" color="text.secondary">
              No sample datasets available. Please regenerate them via `create_demo_datasets.py`.
            </Typography>
          )}
        </Stack>
      </Paper>

      {errorMessage && (
        <Alert severity="error" onClose={onDismissError}>
          {errorMessage}
        </Alert>
      )}
      {statusMessage && !errorMessage && (
        <Alert severity="info" onClose={onDismissStatus}>
          {statusMessage}
        </Alert>
      )}

      <input hidden ref={fileInputRef} type="file" multiple accept=".csv" onChange={onFileChange} />

      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems={{ xs: 'stretch', sm: 'center' }}>
        <Button
          variant="contained"
          startIcon={<CloudUploadIcon />}
          onClick={onSelectFile}
          size="large"
          disabled={isProcessing}
        >
          {isProcessing ? 'Processing...' : 'Choose File(s)'}
        </Button>
        {isProcessing && <CircularProgress size={24} />}
        <Typography variant="body2" color={selectedFileName ? 'text.primary' : 'text.secondary'} sx={{ flexGrow: 1 }}>
          {selectedFileName || 'No file selected'}
        </Typography>
      </Stack>

      <Divider />

      <Stack spacing={1.5}>
        <Button
          variant="contained"
          color="secondary"
          startIcon={<PlayArrowIcon />}
          size="large"
          onClick={onRunModel}
          disabled={!(phaseSummaries.charge && phaseSummaries.discharge) || isProcessing}
        >
          Run Model Analysis
        </Button>
        <Typography variant="caption" color="text.secondary">
          Model inference will run on the backend. This page is not yet connected to the processing pipeline.
        </Typography>
      </Stack>
    </Stack>
  </Paper>
);

export default UploadPanel;

