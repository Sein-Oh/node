const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node-gpu');
const canvas = require('pureimage');

const IMAGE_SIZE = 24;

function GetSubFolder(location) {
    return fs.readdirSync(location, { withFileTypes: true }).filter(dirent => dirent.isDirectory()).map(dirent => dirent.name);
}

function GetFileName(path, filter) {
    return fs.readdirSync(path).filter((file) => file.endsWith(filter));
}

function LoadImage(imgPath) {
    let img;
    if (imgPath.endsWith('.png')) {
        img = canvas.decodePNGFromStream(fs.createReadStream(imgPath));
    }
    else if (imgPath.endsWith('.jpg')) {
        img = canvas.decodeJPEGFromStream(fs.createReadStream(imgPath));
    }
    else {
        console.log('Image format not suppored. You can use jpg or png.')
    }
    return img;
}

function MakeTensor(imgData) {
    const tens = tf.browser.fromPixels(imgData)
        .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE])
        .div(255.0)
        .expandDims();
    return tens;
}

async function makeCNN(inputSize, numClass) {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        activation: 'relu',
        inputShape: [inputSize, inputSize, 3],
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
        units: numClass,
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
    console.log('Loss : %s , Accuracy : %s', logs.loss.toFixed(6), logs.acc.toFixed(6));
}

let model;
async function train() {
    const dataPath = './dataset/train';
    const filter = '.jpg';
    const classArray = GetSubFolder(dataPath);
    const fileArray = classArray.map(className => {
        return GetFileName(path.join(dataPath, className), filter)
    });
    console.log('============================');
    console.log('Found %s class.', classArray.length);
    classArray.forEach(function (name, i) {
        console.log(' - %s : %s EA', name, fileArray[i].length);
    });
    console.log('============================');
    const filePathArray = [];
    const labelArray = [];
    fileArray.forEach((file, index) => {
        file.forEach(f => {
            const fullPath = path.join(dataPath, classArray[index], f);
            filePathArray.push(fullPath);
            labelArray.push(classArray[index]);
        })
    });

    let xs;
    for (let file of filePathArray) {
        const t = await LoadImage(file).then(t => MakeTensor(t));
        if (typeof xs == 'undefined') {
            xs = t;
        } else {
            xs = xs.concat(t);
        }
    }

    const y = [];
    for (let label of labelArray) {
        let index = 0;
        for (let className of classArray) {
            if (label == className) {
                y.push(index);
            }
            index++;
        }
    }
    const ys = tf.oneHot(y, classArray.length);

    model = await makeCNN(IMAGE_SIZE, classArray.length);
    model.summary();
    console.log("Let's start training...")
    await model.fit(xs, ys, {
        stepsPerEpoch: 15,
        epochs: 12,
        callbacks: { onBatchEnd }
    });
    console.log("Complete training.")
}
train();