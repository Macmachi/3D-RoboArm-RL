//Créé par Arnaud R. avec l'aide de Claude 3.5 Sonnet 

let scene, camera, renderer, robot;
const segments = {};
let ball;
const learningRate = 0.1;
const discountFactor = 0.9;
const epsilon = 0.1;

// Variables pour suivre l'apprentissage
let episodeCount = 0;
let totalReward = 0;
let stepCount = 0;
let rewardHistory = [];

// Variable pour contrôler le nombre d'itérations par frame
let iterationsPerFrame = 1;

let ballPositionVariation = 0;
const maxBallPositionVariation = 200;
const variationIncreaseRate = 0.1;

// Initialisation du graphique
let rewardChart;

class NeuralNetwork {
    constructor(inputSize, hiddenSize, outputSize) {
        this.weights1 = new Array(inputSize).fill().map(() => new Array(hiddenSize).fill().map(() => Math.random() - 0.5));
        this.weights2 = new Array(hiddenSize).fill().map(() => new Array(outputSize).fill().map(() => Math.random() - 0.5));
        this.bias1 = new Array(hiddenSize).fill().map(() => Math.random() - 0.5);
        this.bias2 = new Array(outputSize).fill().map(() => Math.random() - 0.5);
    }

    forward(input) {
        let hidden = input.map((_, i) => 
            this.weights1[i].reduce((sum, w, j) => sum + w * input[j], 0)
        ).map((x, i) => Math.max(0, x + this.bias1[i])); // ReLU activation

        let output = this.weights2.map(w => 
            w.reduce((sum, weight, j) => sum + weight * hidden[j], 0)
        ).map((x, i) => x + this.bias2[i]);

        return output;
    }

    train(input, target, learningRate) {
        // Forward pass
        let hidden = input.map((_, i) => 
            this.weights1[i].reduce((sum, w, j) => sum + w * input[j], 0)
        ).map((x, i) => Math.max(0, x + this.bias1[i])); // ReLU activation

        let output = this.weights2.map(w => 
            w.reduce((sum, weight, j) => sum + weight * hidden[j], 0)
        ).map((x, i) => x + this.bias2[i]);

        // Backpropagation
        let outputDelta = output.map((o, i) => o - target[i]);

        let hiddenDelta = this.weights2[0].map((_, i) => 
            this.weights2.reduce((sum, w) => sum + w[i] * outputDelta[i], 0) * (hidden[i] > 0 ? 1 : 0)
        );

        // Update weights and biases
        this.weights2 = this.weights2.map((w, i) => 
            w.map((weight, j) => weight - learningRate * outputDelta[i] * hidden[j])
        );
        this.bias2 = this.bias2.map((b, i) => b - learningRate * outputDelta[i]);

        this.weights1 = this.weights1.map((w, i) => 
            w.map((weight, j) => weight - learningRate * hiddenDelta[j] * input[i])
        );
        this.bias1 = this.bias1.map((b, i) => b - learningRate * hiddenDelta[i]);
    }
}

let nn = new NeuralNetwork(5, 20, 10); // 5 inputs, 20 hidden neurons, 10 outputs (one for each action)
function initRewardChart() {
    const ctx = document.getElementById('rewardChart').getContext('2d');
    rewardChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Récompense moyenne',
                data: [],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function updateRewardChart() {
    if (!rewardChart) return;  // Sortir si rewardChart n'est pas initialisé
    if (rewardHistory.length > 50) {
        rewardHistory.shift();
        rewardChart.data.labels.shift();
    }
    rewardChart.data.labels.push(episodeCount);
    rewardChart.data.datasets[0].data = rewardHistory;
    rewardChart.update();
}


function createBall() {
    const ballGeometry = new THREE.SphereGeometry(15, 32, 32);
    const ballMaterial = new THREE.MeshPhongMaterial({ color: 0xff0000 });
    ball = new THREE.Mesh(ballGeometry, ballMaterial);
    ball.position.set(100, 100, 0);
    scene.add(ball);
}

function getState() {
    return {
        baseRotation: Math.round(segments.base.rotation.y * (180 / Math.PI)),
        shoulderRotation: Math.round(segments.shoulder.rotation.z * (180 / Math.PI)),
        elbowRotation: Math.round(segments.elbow.rotation.z * (180 / Math.PI)),
        wristRotation: Math.round(segments.wrist.rotation.y * (180 / Math.PI)),
        gripperOpen: Math.round(-segments.leftGripper.rotation.z * (180 / Math.PI))
    };
}

function getActions() {
    return [
        { joint: 'base', change: 5 },
        { joint: 'base', change: -5 },
        { joint: 'shoulder', change: 3 },
        { joint: 'shoulder', change: -3 },
        { joint: 'elbow', change: 3 },
        { joint: 'elbow', change: -3 },
        { joint: 'wrist', change: 5 },
        { joint: 'wrist', change: -5 },
        { joint: 'gripper', change: 1 },
        { joint: 'gripper', change: -1 }
    ];
}

function getReward() {
    const gripperPosition = new THREE.Vector3();
    segments.leftGripper.getWorldPosition(gripperPosition);
    const distance = gripperPosition.distanceTo(ball.position);
    
    if (distance < 20 && segments.leftGripper.rotation.z > -0.1) {
        return 100; // La balle est saisie
    } else {
        return -distance; // Récompense négative basée sur la distance
    }
}

function chooseAction(state) {
    if (Math.random() < epsilon) {
        return getActions()[Math.floor(Math.random() * getActions().length)];
    } else {
        const stateArray = Object.values(state);
        const qValues = nn.forward(stateArray);
        const bestActionIndex = qValues.indexOf(Math.max(...qValues));
        return getActions()[bestActionIndex];
    }
}

function trainNN(state, action, nextState, reward) {
    const stateArray = Object.values(state);
    const nextStateArray = Object.values(nextState);
    
    const currentQ = nn.forward(stateArray);
    const nextQ = nn.forward(nextStateArray);
    
    const actionIndex = getActions().findIndex(a => JSON.stringify(a) === JSON.stringify(action));
    const target = [...currentQ];
    target[actionIndex] = reward + discountFactor * Math.max(...nextQ);
    
    nn.train(stateArray, target, learningRate);
}

function executeAction(action) {
    const state = getState();
    switch (action.joint) {
        case 'base':
            state.baseRotation = Math.max(-180, Math.min(180, state.baseRotation + action.change));
            break;
        case 'shoulder':
            state.shoulderRotation = Math.max(-90, Math.min(90, state.shoulderRotation + action.change));
            break;
        case 'elbow':
            state.elbowRotation = Math.max(-135, Math.min(0, state.elbowRotation + action.change));
            break;
        case 'wrist':
            state.wristRotation = Math.max(-180, Math.min(180, state.wristRotation + action.change));
            break;
        case 'gripper':
            state.gripperOpen = Math.max(0, Math.min(15, state.gripperOpen + action.change));
            break;
    }
    updateRobotPose(state);
}

function learn() {
    const currentState = getState();
    const action = chooseAction(currentState);
    executeAction(action);
    const nextState = getState();
    const reward = getReward();
    trainNN(currentState, action, nextState, reward);

    // Mise à jour des statistiques d'apprentissage
    stepCount++;
    totalReward += reward;

    if (reward === 100) {  // La balle a été saisie
        episodeCount++;
        resetEpisode();
    }

    // Mise à jour des statistiques toutes les 100 itérations
    if (stepCount % 100 === 0) {
        updateLearningStats();
    }
}

function updateLearningStats() {
    document.getElementById('episodeCount').textContent = episodeCount;
    const averageReward = totalReward / stepCount;
    document.getElementById('averageReward').textContent = averageReward.toFixed(2);
    rewardHistory.push(averageReward);
    if (rewardChart) updateRewardChart();  // Vérifier si rewardChart existe
}

function resetEpisode() {
    // Réinitialiser la position du bras robotique
    const initialState = {
        baseRotation: 0,
        shoulderRotation: 0,
        elbowRotation: -45,
        wristRotation: 0,
        gripperOpen: 0
        
    };
    updateRobotPose(initialState);

    // Réinitialiser la position de la balle
    if (episodeCount % 100 === 0 && ballPositionVariation < maxBallPositionVariation) {
        ballPositionVariation += variationIncreaseRate;
    }

    ball.position.set(
        100 + (Math.random() * 2 - 1) * ballPositionVariation,
        100 + (Math.random() * 2 - 1) * ballPositionVariation,
        (Math.random() * 2 - 1) * ballPositionVariation
    );
    updateLearningStats();

}

function animate() {
    requestAnimationFrame(animate);
    for (let i = 0; i < iterationsPerFrame; i++) {
        learn();
    }
    renderer.render(scene, camera);
}

function init() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, (window.innerWidth * 0.7) / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 100, 300);
    camera.lookAt(0, 100, 0);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth * 0.7, window.innerHeight);
    renderer.shadowMap.enabled = true;
    document.getElementById('leftPanel').appendChild(renderer.domElement);

    initRewardChart();

    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);

    const pointLight = new THREE.PointLight(0xffffff, 1, 1000);
    pointLight.position.set(0, 200, 100);
    scene.add(pointLight);

    createRobot();
    createBall();
    resetEpisode(); // Ajoutez cette ligne
    animate();

    // Ajout de l'écouteur d'événements pour le slider d'itérations par frame
    document.getElementById('iterationsPerFrame').addEventListener('input', function() {
        iterationsPerFrame = parseInt(this.value);
        document.getElementById('iterationsPerFrameValue').textContent = this.value;
    });
}

function createRobot() {
    robot = new THREE.Group();

    const baseGeometry = new THREE.CylinderGeometry(30, 30, 10, 32);
    const baseMaterial = new THREE.MeshPhongMaterial({ color: 0x4a4a4a });
    segments.base = new THREE.Mesh(baseGeometry, baseMaterial);
    robot.add(segments.base);

    const shoulderGeometry = new THREE.BoxGeometry(20, 20, 20);
    const shoulderMaterial = new THREE.MeshPhongMaterial({ color: 0x6a6a6a });
    segments.shoulder = new THREE.Mesh(shoulderGeometry, shoulderMaterial);
    segments.shoulder.position.set(0, 15, 0);
    segments.base.add(segments.shoulder);

    const upperArmGeometry = new THREE.CylinderGeometry(5, 5, 80, 32);
    const upperArmMaterial = new THREE.MeshPhongMaterial({ color: 0x4a4a4a });
    segments.upperArm = new THREE.Mesh(upperArmGeometry, upperArmMaterial);
    segments.upperArm.position.set(0, 40, 0);
    segments.shoulder.add(segments.upperArm);

    const elbowGeometry = new THREE.SphereGeometry(10, 32, 32);
    const elbowMaterial = new THREE.MeshPhongMaterial({ color: 0x6a6a6a });
    segments.elbow = new THREE.Mesh(elbowGeometry, elbowMaterial);
    segments.elbow.position.set(0, 40, 0);
    segments.upperArm.add(segments.elbow);

    const forearmGeometry = new THREE.CylinderGeometry(4, 4, 60, 32);
    const forearmMaterial = new THREE.MeshPhongMaterial({ color: 0x4a4a4a });
    segments.forearm = new THREE.Mesh(forearmGeometry, forearmMaterial);
    segments.forearm.position.set(0, 30, 0);
    segments.elbow.add(segments.forearm);

    const wristGeometry = new THREE.BoxGeometry(15, 15, 15);
    const wristMaterial = new THREE.MeshPhongMaterial({ color: 0x6a6a6a });
    segments.wrist = new THREE.Mesh(wristGeometry, wristMaterial);
    segments.wrist.position.set(0, 30, 0);
    segments.forearm.add(segments.wrist);

    const gripperBaseGeometry = new THREE.BoxGeometry(20, 10, 5);
    const gripperBaseMaterial = new THREE.MeshPhongMaterial({ color: 0x4a4a4a });
    segments.gripperBase = new THREE.Mesh(gripperBaseGeometry, gripperBaseMaterial);
    segments.gripperBase.position.set(0, 5, 0);
    segments.wrist.add(segments.gripperBase);

    const gripperGeometry = new THREE.BoxGeometry(5, 30, 2);
    const gripperMaterial = new THREE.MeshPhongMaterial({ color: 0x6a6a6a });
    segments.leftGripper = new THREE.Mesh(gripperGeometry, gripperMaterial);
    segments.leftGripper.position.set(-5, 15, 0);
    segments.gripperBase.add(segments.leftGripper);

    segments.rightGripper = segments.leftGripper.clone();
    segments.rightGripper.position.set(5, 15, 0);
    segments.gripperBase.add(segments.rightGripper);

    scene.add(robot);
}

function updateRobotPose(state) {
    segments.base.rotation.y = THREE.MathUtils.degToRad(state.baseRotation);
    segments.shoulder.rotation.z = THREE.MathUtils.degToRad(state.shoulderRotation);
    segments.elbow.rotation.z = THREE.MathUtils.degToRad(state.elbowRotation);
    segments.wrist.rotation.y = THREE.MathUtils.degToRad(state.wristRotation);
    
    let gripperOpen = state.gripperOpen;
    
    segments.leftGripper.rotation.z = THREE.MathUtils.degToRad(-gripperOpen);
    segments.rightGripper.rotation.z = THREE.MathUtils.degToRad(gripperOpen);

    // Mise à jour des sliders
    document.getElementById('baseRotation').value = state.baseRotation;
    document.getElementById('shoulderRotation').value = state.shoulderRotation;
    document.getElementById('elbowRotation').value = state.elbowRotation;
    document.getElementById('wristRotation').value = state.wristRotation;
    document.getElementById('gripperOpen').value = gripperOpen;
}

window.addEventListener('resize', () => {
    camera.aspect = (window.innerWidth * 0.7) / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth * 0.7, window.innerHeight);
    rewardChart.resize();
});

document.addEventListener('DOMContentLoaded', function() {
    init();
});