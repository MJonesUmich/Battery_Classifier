import { Alert, Box, Container, Stack, Typography } from '@mui/material';
import { useCallback, useMemo, useRef, useState } from 'react';
import './App.css';
import EnterpriseCta from './components/EnterpriseCta';
import HeroBanner from './components/HeroBanner';
import HowItWorksCard from './components/HowItWorksCard';
import PredictionPreview from './components/PredictionPreview';
import UploadPanel from './components/UploadPanel';
import useSampleDatasets from './hooks/useSampleDatasets';
import logo from './logo.svg';
import predictChemistry from './utils/logregPredict';
import { buildFeatureMap, mergePhaseSummaries, parseCsvFile, parseCsvString } from './utils/phaseProcessing';

function App() {
  const fileInputRef = useRef(null);
  const [selectedFileName, setSelectedFileName] = useState('');
  const [phaseSummaries, setPhaseSummaries] = useState({ charge: null, discharge: null });
  const [prediction, setPrediction] = useState(null);
  const [probabilities, setProbabilities] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);

  const datasetBaseUrl = useMemo(() => `${process.env.PUBLIC_URL || ''}/datasets`, []);
  const { sampleDatasets, datasetsLoading, datasetsError } = useSampleDatasets(datasetBaseUrl);

  const handleSelectFile = () => {
    fileInputRef.current?.click();
  };

  const runPrediction = useCallback((summaries) => {
      const featureMap = buildFeatureMap(summaries.charge?.stats, summaries.discharge?.stats);
      const result = predictChemistry(featureMap);
      setPrediction(result.label);
      setProbabilities(result.probabilities);
      setStatusMessage('Prediction generated using uploaded charge & discharge files.');
      setErrorMessage('');
  }, []);

  const handleFileChange = async (event) => {
    const files = Array.from(event.target.files || []);
    if (!files.length) {
      return;
    }
    setSelectedFileName(files.map((file) => file.name).join(', '));
    setIsProcessing(true);
    setErrorMessage('');
    setStatusMessage('Parsing uploaded file(s)...');

    let updatedSummaries = { ...phaseSummaries };
    try {
      for (const file of files) {
        const rows = await parseCsvFile(file);
        if (!rows.length) {
          throw new Error(`File ${file.name} does not contain any rows.`);
        }
        updatedSummaries = mergePhaseSummaries(rows, file.name, updatedSummaries);
      }
      setPhaseSummaries(updatedSummaries);
      if (updatedSummaries.charge && updatedSummaries.discharge) {
        setStatusMessage('Charge & discharge ready. Click "Run Model Analysis" to generate prediction.');
        setPrediction(null);
        setProbabilities(null);
      } else {
        setStatusMessage('Upload both charge and discharge CSV files to run the prediction.');
        setPrediction(null);
        setProbabilities(null);
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setIsProcessing(false);
    }
  };

  const handleSamplePredict = async (dataset) => {
    if (!dataset?.url) return;
    try {
      setIsProcessing(true);
      setStatusMessage(`Loading sample "${dataset.title}"...`);
      setErrorMessage('');

      const response = await fetch(dataset.url);
      if (!response.ok) {
        throw new Error(`Failed to fetch sample CSV (${response.status})`);
      }
      const text = await response.text();
      const rows = parseCsvString(text);

      const updatedSummaries = mergePhaseSummaries(rows, dataset.file, {});
      setPhaseSummaries(updatedSummaries);
      setSelectedFileName(`${dataset.file} (sample)`);

      if (updatedSummaries.charge && updatedSummaries.discharge) {
        setStatusMessage('Sample ready. Click "Run Model Analysis" to generate prediction.');
        setPrediction(null);
        setProbabilities(null);
      } else {
        setStatusMessage('Sample is missing charge or discharge data.');
        setPrediction(null);
        setProbabilities(null);
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setIsProcessing(false);
    }
  };

  const handleRunModel = () => {
    if (!phaseSummaries.charge || !phaseSummaries.discharge) {
      setStatusMessage('Upload both charge and discharge CSV files to run the prediction.');
      return;
    }
    try {
      setStatusMessage('Running model...');
      runPrediction(phaseSummaries);
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : String(err));
    }
  };

  const probabilityEntries = useMemo(() => {
    if (!probabilities) return [];
    return Object.entries(probabilities).map(([label, value]) => ({
      label,
      percent: (value * 100).toFixed(2),
    }));
  }, [probabilities]);

  const topProbability = useMemo(() => {
    if (!prediction || !probabilities) return null;
    const value = probabilities[prediction];
    return typeof value === 'number' ? value * 100 : null;
  }, [prediction, probabilities]);

  return (
    <Box className="app-root">
      <Container maxWidth="md" sx={{ py: { xs: 6, md: 8 } }}>
        <Stack spacing={4}>
          <HeroBanner logoSrc={logo} />

          <Alert severity="info" sx={{ alignItems: 'center' }}>
            <Typography variant="body2">
              You are exploring the interactive trial experience. Once you validate the workflow, switch to our managed
              batch update pipeline or request direct API access for large-scale inference.
            </Typography>
          </Alert>

          <HowItWorksCard />

          <UploadPanel
            fileInputRef={fileInputRef}
            selectedFileName={selectedFileName}
            isProcessing={isProcessing}
            sampleDatasets={sampleDatasets}
            datasetsLoading={datasetsLoading}
            datasetsError={datasetsError}
            phaseSummaries={phaseSummaries}
            errorMessage={errorMessage}
            statusMessage={statusMessage}
            onSelectFile={handleSelectFile}
            onFileChange={handleFileChange}
            onSamplePredict={handleSamplePredict}
            onRunModel={handleRunModel}
            onDismissError={() => setErrorMessage('')}
            onDismissStatus={() => setStatusMessage('')}
          />

          <PredictionPreview prediction={prediction} topProbability={topProbability} probabilityEntries={probabilityEntries} />

          <EnterpriseCta />
        </Stack>
      </Container>
    </Box>
  );
}

export default App;
