
let canvas, ctx;

let mouseX, mouseY, mouseDown = 0;

let touchX, touchY;

let lastX, lastY = -1;

function drawLine(ctx, x, y, size) {

    if (lastX == -1) {
        lastX = x;
        lastY = y;
    }

    r = 255; g = 255; b = 255; a = 255;

    ctx.strokeStyle = "rgba(" + r + "," + b + "," + (a / 255) + ")";

    ctx.linecap = "round";

    ctx.beginPath();

    ctx.moveTo(lastX, lastY);

    ctx.lineTo(x, y);
    ctx.stroke();

    ctx.closePath();

    lastX = x;
    lastY = y;
}

function clearCanvas(canvas, ctx) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('canvasarea').innerHTML = '';
}

function sketchpad_mouseDown() {
    mouseDown = 1;
    drawLine(ctx, mouseX, mouseY, 12);
}

function sketchpad_mouseUp() {
    mouseDown = 0;

    lastX = -1;
    lastY = -1;
}

function sketchpad_mouseMove(e) {

    getMousePos(e);

    if (mouseDown == 1) {
        drawLine(ctx, mouseX, mouseY, 12);
    }
}

function getMousePos(e) {
    if (!e) {
        let e = event;
    }

    if (e.offsetX) {
        mouseX = e.offsetX;
        mouseY = e.offsetY;
    } else if (e.layerX) {
        mouseX = e.layerX;
        mouseY = e.layerY;
    }
}

function sketchpad_touchStart() {

    getTouchPos();

    drawLine(ctx, touchX, touchY, 12);

    event.preventDefault();
}

function sketchpad_touchEnd() {
    lastX = -1;
    lastY = -1;
}

function sketchpad_touchMove(e) {

    getTouchPos();

    drawLine(ctx, touchX, touchY, 12);

    event.preventDefault();
}

function getTouchPos(e) {

    if (!e) {
        let e = event;
    }

    if (e.touches) {
        if (e.touches.length == 1) {
            let touch = e.touches[0];
            touchX = touch.pageX - touch.target.offsetLeft;
            touchY = touch.pageY - touch.target.offsetTop;
        }
    }
}

async function init() {

    canvas = document.getElementById('canvas');

    if (canvas.getContext) {
        ctx = canvas.getContext('2d');
    }

    if (ctx) {

        canvas.addEventListener('mousedown', sketchpad_mouseDown, false);
        canvas.addEventListener('mousemove', sketchpad_mouseMove, false);
        window.addEventListener('mouseup', sketchpad_mouseUp, false);

        canvas.addEventListener('touchstart', sketchpad_touchStart, false);
        canvas.addEventListener('touchend', sketchpad_touchEnd, false);
        canvas.addEventListener('touchmove', sketchpad_mouseUp, false);
    }

    model = await tf.loadModel('../saved_models/model.json')
}

function predict() {

    const imageData = ctx.getImageData(0, 0, 140, 140);

    let tfImg = tf.fromPixels(imageData, 1);
    let smallImg = tf.image.resizeBilinear(tfImg, [28, 28]);
    smallImg = tf.cast(smallImg, 'float32');
    let tensor = smallImg.expandDims(0);
    tensor = tensor.div(tf.scalar(255));
    const prediction = model.predict(tensor);
    const predictionValues = prediction.dataSync();

    let isThereAnyPrediction = false;
    for (index = 0; index < predictedValues.length; index++) {
        if (predictionValues[index] > 0.5) {
            isThereAnyPrediction = true;
            document.getElementById('blankspace').innerHTML = '<br/>Predicted Number: ' + index;
        }
    }

    if (!isThereAnyPrediction) {
        document.getElementById('blankspace').innerHTML = '<br/>Unable to Predict';
    }
}