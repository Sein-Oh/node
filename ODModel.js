const tf = require("@tensorflow/tfjs");
const LABEL_MULTIPLIER = [224, 1, 1, 1, 1];
const topLayerGroupNames = ['conv_pw_9', 'conv_pw_10', 'conv_pw_11'];

    // Name of the layer that will become the top layer of the truncated base.
    const topLayerName =
        `${topLayerGroupNames[topLayerGroupNames.length - 1]}_relu`;
    async function buildObjectDetectionModel() {
        const { truncatedBase, fineTuningLayers } = await loadTruncatedBase();

        // Build the new head model.
        const newHead = buildNewHead(truncatedBase.outputs[0].shape.slice(1));
        const newOutput = newHead.apply(truncatedBase.outputs[0]);
        const model = tf.model({ inputs: truncatedBase.inputs, outputs: newOutput });

        return { model, fineTuningLayers };
    }

    async function loadTruncatedBase() {
        // TODO(cais): Add unit test.
        const mobilenet = await tf.loadLayersModel(
            'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');

        // Return a model that outputs an internal activation.
        const fineTuningLayers = [];
        const layer = mobilenet.getLayer(topLayerName);
        const truncatedBase =
            tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
        // Freeze the model's layers.
        for (const layer of truncatedBase.layers) {
            layer.trainable = false;
            for (const groupName of topLayerGroupNames) {
                if (layer.name.indexOf(groupName) === 0) {
                    fineTuningLayers.push(layer);
                    break;
                }
            }
        }

        tf.util.assert(
            fineTuningLayers.length > 1,
            `Did not find any layers that match the prefixes ${topLayerGroupNames}`);
        return { truncatedBase, fineTuningLayers };
    }

    function buildNewHead(inputShape) {
        const newHead = tf.sequential();
        newHead.add(tf.layers.flatten({ inputShape }));
        newHead.add(tf.layers.dense({ units: 200, activation: 'relu' }));
        // Five output units:
        //   - The first is a shape indictor: predicts whether the target
        //     shape is a triangle or a rectangle.
        //   - The remaining four units are for bounding-box prediction:
        //     [left, right, top, bottom] in the unit of pixels.
        newHead.add(tf.layers.dense({ units: 5 }));
        return newHead;
    }

    function customLossFunction(yTrue, yPred) {
        return tf.tidy(() => {
            // Scale the the first column (0-1 shape indicator) of `yTrue` in order
            // to ensure balanced contributions to the final loss value
            // from shape and bounding-box predictions.
            return tf.metrics.meanSquaredError(yTrue.mul(LABEL_MULTIPLIER), yPred);
        });
    }

    async function load() {
        const { model, fineTuningLayers } = await buildObjectDetectionModel();
        await model.compile({loss: customLossFunction, optimizer: tf.train.rmsprop(2e-3)});
        //model.summary();
        return { model, fineTuningLayers };
    }

    module.exports.load = load;