function getFile(path, filter){
    return fs.readdirSync(path).filter((file) => file.endsWith(filter));
}

function getFolder(path){
    return fs.readdirSync(path, {withFileTypes:true}).filter(dirent => dirent.isDirectory()).map(dirent => dirent.name);
}
