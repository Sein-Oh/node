const PImage = require('pureimage');
const fs = require('fs');

PImage.decodeJPEGFromStream(fs.createReadStream("rose.jpg")).then((img) => {
    console.log("size is",img.width,img.height);
    var img2 = PImage.make(84,84);
    var c = img2.getContext('2d');
    c.drawImage(img,
        0, 0, img.width, img.height, // source dimensions
        0, 0, 84, 84                 // destination dimensions
    );
    PImage.encodeJPEGToStream(img2,fs.createWriteStream('resize_rose.jpg')).then(() => {
        console.log("done writing");
    });
});