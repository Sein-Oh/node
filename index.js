const fs = require('fs');
const canvas = require('pureimage');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');

let IMAGE_SIZE = 224;
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
    LABEL.forEach(function(className, index) {
        if(label == className){
            (yMerge == undefined) ? yMerge = tf.oneHot(index, LABEL.length) : yMerge = yMerge.concat(tf.oneHot(index, LABEL.length));
        }
    });
    return yMerge;
}

async function run() {
    const a = './dataset/train';
    const f = 'png';
    const imgPath = await getPath(a, f);
    let xs, ys;
    for (let p of imgPath) {
        xs = await loadImage(p[1]).then(img => makeTensor(img)).then(tensor => mergeTensor(tensor));
        ys = await mergeLabel(p[0]);
    }
    console.log(xs.shape);
    console.log(ys.shape);
}

run();
