const fs = require('fs');
const path = require('path');

function makeDataset(dataPath, filter){
    const label = fs.readdirSync(dataPath, {withFileTypes:true}).filter(dirent => dirent.isDirectory()).map(dirent => dirent.name);
    console.log('found %s label.', label.length);
    const fileArray = [];
    label.forEach(e=> {
        console.log('-', e);
        const imgPath = path.join(dataPath, e);
        const imgName = fs.readdirSync(imgPath).filter((file) => file.endsWith(filter));
        imgName.forEach(name =>{
            fileArray.push([e, name]);
        })
    });
    console.log(fileArray);
}

const train = './dataset/train'
const filter = '.png';
makeDataset(train, filter);


function makeGen(dataPath, filter){
    fs.readdir(dataPath, {withFileTypes:true}, function(err, data){
        const label = data.filter(dirent => dirent.isDirectory()).map(dirent => dirent.name);
        console.log(label);
    });
}
makeGen(train, filter);
