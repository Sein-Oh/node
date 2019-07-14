const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const canvas = require('pureimage');

const IMAGE_SIZE = 224;

function makeCNN(inputSize, numClass) {
    return new Promise(function (resolve, reject) {
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
        resolve(model);
    });
}

//클래스와 이미지 경로를 만든다.
function getPath(dataPath, filter) {
    return new Promise(function (resolve, reject) {
        const numClass = fs.readdirSync(dataPath, { withFileTypes: true }).filter(dirent => dirent.isDirectory()).map(dirent => dirent.name);
        console.log('Pound %s class', numClass.length);
        const class_path = [];
        numClass.forEach(className => {
            const filePath = path.join(dataPath, className);
            const files = fs.readdirSync(filePath).filter((file) => file.endsWith(filter));
            console.log(' - %s : %s items.', className, files.length);
            files.forEach(async (f, i) => {
                const fullPath = path.join(dataPath, className, f);
                class_path.push([className, fullPath]);
            });
        });
        resolve(class_path);
    });
}

function makeTensor(imgData) {
    return new Promise(function (resolve, reject) {
        const tens = tf.browser.fromPixels(imgData)
            .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE])
            .div(255.0)
            .expandDims();
        resolve(tens);
    });
}

function LoadImage(imgPath) {
    return new Promise(function (resolve, reject) {
        if (imgPath.endsWith('.png')) {
            img = canvas.decodePNGFromStream(fs.createReadStream(imgPath));
        }
        else if (imgPath.endsWith('.jpg')) {
            img = canvas.decodeJPEGFromStream(fs.createReadStream(imgPath));
        }
        else {
            console.log('Image format not suppored. You can use jpg or png.')
        }
        resolve(img);
    });
}
