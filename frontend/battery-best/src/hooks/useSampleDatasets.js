import { useEffect, useState } from 'react';

const useSampleDatasets = (datasetBaseUrl) => {
  const [sampleDatasets, setSampleDatasets] = useState([]);
  const [datasetsLoading, setDatasetsLoading] = useState(true);
  const [datasetsError, setDatasetsError] = useState('');

  useEffect(() => {
    const controller = new AbortController();

    const loadManifest = async () => {
      try {
        setDatasetsLoading(true);
        const response = await fetch(`${datasetBaseUrl}/datasets.json`, { signal: controller.signal });
        if (!response.ok) {
          throw new Error(`Failed to load datasets.json (${response.status})`);
        }
        const payload = await response.json();
        const entries = (payload.datasets || []).map((item) => ({
          ...item,
          url: `${datasetBaseUrl}/${item.file}`,
        }));
        setSampleDatasets(entries);
        setDatasetsError('');
      } catch (err) {
        if (err.name !== 'AbortError') {
          setDatasetsError(err.message || 'Unable to load sample datasets.');
        }
      } finally {
        setDatasetsLoading(false);
      }
    };

    loadManifest();
    return () => controller.abort();
  }, [datasetBaseUrl]);

  return { sampleDatasets, datasetsLoading, datasetsError };
};

export default useSampleDatasets;

