import modelBundle from "../assets/logreg_model.json";

const EPSILON = 1e-12;

const softmax = (logits) => {
  const maxLogit = Math.max(...logits);
  const exp = logits.map((value) => Math.exp(value - maxLogit));
  const sum = exp.reduce((acc, value) => acc + value, 0) + EPSILON;
  return exp.map((value) => value / sum);
};

const standardize = (features, names, scaler) =>
  names.map((name, idx) => {
    if (!(name in features)) {
      throw new Error(`Missing feature: ${name}`);
    }
    const raw = Number(features[name]);
    if (Number.isNaN(raw)) {
      throw new Error(`Feature ${name} is not numeric.`);
    }
    const mean = scaler.mean[idx];
    const scale = scaler.scale[idx] || 1;
    return (raw - mean) / scale;
  });

export const featureNames = modelBundle.feature_names;

export function predictChemistry(
  featureInput,
  model = modelBundle
) {
  if (!featureInput) {
    throw new Error("featureInput is required.");
  }

  const standardized = standardize(
    featureInput,
    model.feature_names,
    model.scaler
  );

  const logits = model.coef.map((weights, classIdx) => {
    const linear = weights.reduce(
      (acc, weight, featIdx) => acc + weight * standardized[featIdx],
      model.intercept[classIdx]
    );
    return linear;
  });

  const probabilities = softmax(logits);
  const bestIdx = probabilities.indexOf(Math.max(...probabilities));
  const label = model.classes[bestIdx];

  const probabilityMap = model.classes.reduce((acc, cls, idx) => {
    acc[cls] = probabilities[idx];
    return acc;
  }, {});

  return {
    label,
    probability: probabilityMap[label],
    probabilities: probabilityMap,
    logits,
  };
}

export default predictChemistry;


