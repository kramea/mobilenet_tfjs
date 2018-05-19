/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// import * as tf from '@tensorflow/tfjs';

// import {ControllerDataset} from './controller_dataset';
// import * as ui from './ui';
// import {Webcam} from './webcam';

// The number of classes we want to predict. In this example, we will be
// predicting 4 classes for up, down, left, and right.

const MOBILENET_MODEL_PATH =
    // tslint:disable-next-line:max-line-length
    'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

const MODEL_URL = 'tensorflowjs_model.pb';
const WEIGHTS_URL = 'weights_manifest.json';

IMAGE_SIZE = 224;
TOPK_PREDICTIONS = 1;

// A webcam class that generates Tensors from the images from the webcam.
const webcam = new Webcam(document.getElementById('webcam'));
const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;
// The dataset object where we will store activations.
// const controllerDataset = new ControllerDataset(NUM_CLASSES);

const pred = document.getElementById('pred')

let mobilenet;
let isPredicting;
const mobilenetDetect = async () => {
  status('Loading model...');

  mobilenet = await tf_converter.loadFrozenModel(MODEL_URL, WEIGHTS_URL);
  await webcam.setup();

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  // mobilenet.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();
  ui.init();
  // mobilenet.predict(webcam.capture()).dispose();
  const logits = mobilenet.predict(webcam.capture());
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  pred.innerText = classes[0].className;

  status('');

  
  isPredicting = true;
  continuous_prediction();
};


async function continuous_prediction() {

  while (isPredicting) {

    const logits = mobilenet.predict(webcam.capture());
    const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
    pred.innerText = classes[0].className;
    await tf.nextFrame();

  }
}




/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from MobileNet.
 * @param topK The number of top predictions to show.
 */
async function getTopKClasses(logits, topK) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: IMAGENET_CLASSES[topkIndices[i]],
      probability: topkValues[i]
    })
  }
  return topClassesAndProbs;
}




mobilenetDetect();



