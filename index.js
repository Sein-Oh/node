const fs = require('fs');
const canvas = require('pureimage');
const tf = require('@tensorflow/tfjs');

function loadImage(imgPath) {
    return new Promise(function(resolve, reject){
        (imgPath.endsWith('png')) ? img = canvas.decodePNGFromStream(fs.createReadStream(imgPath)) :
        (imgPath.endsWith('jpg')) ? img = canvas.decodeJPEGFromStream(fs.createReadStream(imgPath)) : console.log('Image format not suppored. You can use jpg or png.');
        resolve(img);
    }); 
}

function makeTensor(imgData) {
    return new Promise(function(resolve, reject){
        const tens = tf.browser.fromPixels(imgData)
        .resizeNearestNeighbor([224, 224])
        .div(255.0)
        .expandDims();
        resolve(tens);
    });
};

let xs;
function mergeTensor(tensor) {
    return new Promise(function(resolve, reject){
        (xs == undefined) ? xs = tensor : xs = xs.concat(tensor);
        resolve(xs);
    });
}

const p = ['./dataset/train/circle/circle001.png', './dataset/train/circle/circle002.png']
p.forEach(p => {
    loadImage(p)
        .then(img => makeTensor(img))
        .then(t => mergeTensor(t))
        .then(e => console.log(e.shape));
});
