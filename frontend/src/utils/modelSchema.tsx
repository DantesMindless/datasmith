export type ConfigField = {
  key: string;
  label: string;
  type: "number" | "text" | "select" | "layers";
  default?: any;
  helperText?: string;
  options?: { label: string; value: any }[];

};

export const modelConfigSchemas: Record<string, ConfigField[]> = {
  logistic_regression: [
    { key: "max_iter", label: "Max Iterations", type: "number", default: 100, helperText: "Number of full passes over data" },
    {
      key: "solver", label: "Solver", type: "select", default: "lbfgs", helperText: "Optimization algorithm",
      options: [
        { label: "lbfgs", value: "lbfgs" },
        { label: "saga", value: "saga" },
        { label: "liblinear", value: "liblinear" },
      ],
    },
    { key: "C", label: "Regularization Strength (C)", type: "number", default: 1.0 },
  ],

  random_forest: [
    { key: "n_estimators", label: "Number of Trees", type: "number", default: 100 },
    { key: "max_depth", label: "Max Depth", type: "number", default: 5 },
    { key: "min_samples_split", label: "Min Samples Split", type: "number", default: 2 },
  ],

  svm: [
    {
      key: "kernel", label: "Kernel", type: "select", default: "rbf",
      options: [
        { label: "RBF", value: "rbf" },
        { label: "Linear", value: "linear" },
        { label: "Poly", value: "poly" },
      ],
    },
    { key: "C", label: "Regularization (C)", type: "number", default: 1.0 },
    { key: "gamma", label: "Gamma", type: "text", default: "scale" },
  ],
  naive_bayes: [
  {
    key: "nb_type",
    label: "Naive Bayes Type",
    type: "select",
    default: "gaussian",
    options: [
      { label: "GaussianNB", value: "gaussian" },
      { label: "MultinomialNB", value: "multinomial" },
      { label: "BernoulliNB", value: "bernoulli" },
    ],
    helperText: "Select the variant of Naive Bayes",
  },
  {
    key: "var_smoothing",
    label: "Var Smoothing",
    type: "number",
    default: 1e-9,
    helperText: "For GaussianNB: portion of largest variance added to variances",
  },
  {
    key: "alpha",
    label: "Alpha",
    type: "number",
    default: 1.0,
    helperText: "Additive smoothing parameter for Multinomial/Bernoulli",
  },
  {
    key: "fit_prior",
    label: "Fit Prior",
    type: "select",
    default: true,
    options: [
      { label: "True", value: true },
      { label: "False", value: false },
    ],
    helperText: "Whether to learn class prior probabilities",
  },
  {
    key: "binarize",
    label: "Binarize",
    type: "number",
    default: 0.0,
    helperText: "Threshold for binarizing features (BernoulliNB only)",
  },
],
  knn: [
    { key: "n_neighbors", label: "Neighbors", type: "number", default: 5 },
    {
      key: "weights", label: "Weights", type: "select", default: "uniform",
      options: [
        { label: "Uniform", value: "uniform" },
        { label: "Distance", value: "distance" },
      ],
    },
  ],

  GRADIENT_BOOSTING: [
    { key: "n_estimators", label: "Estimators", type: "number", default: 100 },
    { key: "learning_rate", label: "Learning Rate", type: "number", default: 0.1 },
    { key: "max_depth", label: "Max Depth", type: "number", default: 3 },
  ],

  neural_network: [
    { key: "epochs", label: "Epochs", type: "number", default: 10, helperText: "Number of passes through the dataset" },
    { key: "learning_rate", label: "Learning Rate", type: "number", default: 0.001 },
    { key: "batch_size", label: "Batch Size", type: "number", default: 32 },
    {
      key: "optimizer", label: "Optimizer", type: "select", default: "adam",
      options: [
        { label: "Adam", value: "adam" },
        { label: "SGD", value: "sgd" },
        { label: "RMSprop", value: "rmsprop" },
      ],
    },
    {
      key: "loss_function", label: "Loss Function", type: "select", default: "cross_entropy",
      options: [
        { label: "Cross Entropy", value: "cross_entropy" },
        { label: "MSE", value: "mse" },
      ],
    },
    {
      key: "dense_layers",
      label: "Dense Layers",
      type: "layers",
      helperText: "Configure each dense layer individually",
      default: [
        { units: 128, activation: "relu" },
        { units: 64, activation: "relu" },
        { units: 10, activation: "softmax" },
      ],
    },
    {
      key: "dropout_layers",
      label: "Dropout Layers",
      type: "layers",
      helperText: "Configure dropout layers (optional)",
      default: [
        { rate: 0.2 },
        { rate: 0.5 },
      ],
    },
    {
      key: "use_batch_norm",
      label: "Use Batch Norm",
      type: "select",
      default: true,
      options: [
        { label: "True", value: true },
        { label: "False", value: false },
      ],
    },
    { key: "weight_decay", label: "Weight Decay", type: "number", default: 0.0 },
  ],

  cnn: [
    { key: "epochs", label: "Epochs", type: "number", default: 20 },
    { key: "learning_rate", label: "Learning Rate", type: "number", default: 0.001 },
    { key: "batch_size", label: "Batch Size", type: "number", default: 32 },
    {
      key: "optimizer", label: "Optimizer", type: "select", default: "adam",
      options: [
        { label: "Adam", value: "adam" },
        { label: "SGD", value: "sgd" },
        { label: "RMSprop", value: "rmsprop" },
      ],
    },
    {
      key: "loss_function", label: "Loss Function", type: "select", default: "cross_entropy",
      options: [
        { label: "Cross Entropy", value: "cross_entropy" },
        { label: "MSE", value: "mse" },
      ],
    },
    {
      key: "conv_layers",
      label: "Convolution Layers",
      type: "layers",
      helperText: "Configure convolution layers with filters, kernel sizes, etc.",
      default: [
        { filters: 32, kernel_size: 3, stride: 1, activation: "relu" },
        { filters: 64, kernel_size: 3, stride: 1, activation: "relu" },
      ],
    },
    {
      key: "pooling_layers",
      label: "Pooling Layers",
      type: "layers",
      helperText: "Configure pooling layers after convolutions",
      default: [
        { type: "max", pool_size: 2 },
      ],
    },
    {
      key: "dropout_layers",
      label: "Dropout Layers",
      type: "layers",
      helperText: "Configure dropout layers (optional)",
      default: [
        { rate: 0.25 },
      ],
    },
    {
      key: "use_batch_norm",
      label: "Use Batch Norm",
      type: "select",
      default: true,
      options: [
        { label: "True", value: true },
        { label: "False", value: false },
      ],
    },
  ],
};