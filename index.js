const fs = require('fs');
const canvas = require('pureimage');
const tf = require('@tensorflow/tfjs');
const path = require('path');

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
                class_path.push(fullPath);
            });
        });
        resolve(class_path);
    });
}

function loadImage(imgPath) {
    return new Promise(function (resolve, reject) {
        (imgPath.endsWith('png')) ? img = canvas.decodePNGFromStream(fs.createReadStream(imgPath)) :
            (imgPath.endsWith('jpg')) ? img = canvas.decodeJPEGFromStream(fs.createReadStream(imgPath)) : console.log('Image format not suppored. You can use jpg or png.');
        resolve(img);
    });
}

function makeTensor(imgData) {
    return new Promise(function (resolve, reject) {
        const tens = tf.browser.fromPixels(imgData)
            .resizeNearestNeighbor([224, 224])
            .div(255.0)
            .expandDims();
        resolve(tens);
    });
};

let xs;
function mergeTensor(tensor) {
    return new Promise(function (resolve, reject) {
        (xs == undefined) ? xs = tensor : xs = xs.concat(tensor);
        resolve(xs);
    });
}


const a = './dataset/train';
const f = 'png';
getPath(a,f)
    .then(path => {
        path.forEach(p => {
            loadImage(p)
                .then(img => makeTensor(img))
                .then(tensor => mergeTensor(tensor))
                .then(e => console.log(e.shape));
        });
    });
