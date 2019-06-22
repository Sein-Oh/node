const fs = require('fs');
const path = './';
/*
fs.readdir(path, function(err, files){
    if(err) throw err;
    files.forEach(function(file){
        console.log(path+file);
    })
})
*/

const exec = require('child_process').exec;

exec('mkdir hello');