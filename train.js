const fs = require('fs');
const canvas = require('pureimage');
const tf = require('@tensorflow/tfjs-node-gpu');
const path = require('path');

let IMAGE_SIZE = 32;
let LABEL;

function getPath(dataPath, filter) {
    LABEL = fs.readdirSync(dataPath, { withFileTypes: true }).filter(dirent => dirent.isDirectory()).map(dirent => dirent.name);
    console.log('Pound %s class', LABEL.length);
    const class_path = [];
    LABEL.forEach(className => {
        const filePath = path.join(dataPath, className);
        const files = fs.readdirSync(filePath).filter((file) => file.endsWith(filter));
        console.log(' - %s : %s items.', className, files.length);
        files.forEach(async (f, i) => {
            const fullPath = path.join(dataPath, className, f);
            class_path.push([className, fullPath]);
        });
    });
    return class_path;
}

function loadImage(imgPath) {
    (imgPath.endsWith('png')) ? img = canvas.decodePNGFromStream(fs.createReadStream(imgPath)) :
        (imgPath.endsWith('jpg')) ? img = canvas.decodeJPEGFromStream(fs.createReadStream(imgPath)) : console.log('Image format not suppored. You can use jpg or png.');
    return img;
}

function makeTensor(imgData) {
    const tens = tf.browser.fromPixels(imgData)
        .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE])
        .div(255.0)
        .expandDims();
    return tens;
};

let xMerge;
function mergeTensor(tensor) {
    (xMerge == undefined) ? xMerge = tensor : xMerge = xMerge.concat(tensor);
    return xMerge;
}

let yMerge;
function mergeLabel(label) {
    LABEL.forEach(function (className, index) {
        if (label == className) {
            (yMerge == undefined) ? yMerge = tf.oneHot(index, LABEL.length).expandDims() : yMerge = yMerge.concat(tf.oneHot(index, LABEL.length).expandDims());
        }
    });
    return yMerge;
}

function makeCNN() {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        activation: 'relu',
        inputShape: [IMAGE_SIZE, IMAGE_SIZE, 3],
        strides: 1,
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        activation: 'relu'
    }));
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2]
    }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({
        units: 128,
        activation: 'relu'
    }));
    model.add(tf.layers.dense({
        units: LABEL.length,
        activation: 'softmax'
    }));
    model.compile({
        loss: 'categoricalCrossentropy',
        optimizer: tf.train.adam(),
        metrics: ['accuracy']
    });
    return model;
}

function onBatchEnd(batch, logs) {
    console.log('Loss : %s , Accuracy : %s', logs.loss.toFixed(4), logs.acc.toFixed(4));
}

async function run() {
    const a = './dataset';
    const f = 'png';
    const imgPath = await getPath(a, f);
    let xs, ys;
    for (let p of imgPath) {
        xs = await loadImage(p[1]).then(img => makeTensor(img)).then(tensor => mergeTensor(tensor));
        ys = await mergeLabel(p[0]);
    }
    const model = await makeCNN();
    //model.summary();
    await model.fit(xs, ys, {
        stepsPerEpoch: 500,
        epochs: 10,
        callbacks: { onBatchEnd }
    });
    console.log("Complete training.")
    await model.save('file://saved');

}

run();
