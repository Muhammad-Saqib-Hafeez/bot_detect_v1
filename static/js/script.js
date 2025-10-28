// Real-time human activity simulation
let mouseCount = 0;
let clickCount = 0;
let scrollCount = 0;
let keyCount = 0;

document.addEventListener('mousemove', () => {
    mouseCount++;
    document.getElementById('mouseMovements').textContent = mouseCount;
});

document.addEventListener('click', () => {
    clickCount++;
    document.getElementById('clicks').textContent = clickCount;
});

document.addEventListener('scroll', () => {
    scrollCount++;
    document.getElementById('scrolls').textContent = scrollCount;
});

document.addEventListener('keydown', () => {
    keyCount++;
    document.getElementById('keystrokes').textContent = keyCount;
});

// Enhanced simulation function
function simulateHumanActivity() {
    const simulation = document.getElementById('activitySimulation');
    simulation.style.display = 'block';
    simulation.scrollIntoView({ behavior: 'smooth' });
    
    let simMouseCount = 0;
    let simKeyCount = 0;
    let simScrollCount = 0;
    let simTimingCount = 0;
    
    const mouseInterval = setInterval(() => {
        simMouseCount++;
        document.getElementById('mouseCount').textContent = simMouseCount;
    }, 100);
    
    const keyInterval = setInterval(() => {
        simKeyCount++;
        document.getElementById('keyboardCount').textContent = simKeyCount;
    }, 500);
    
    const scrollInterval = setInterval(() => {
        simScrollCount++;
        document.getElementById('scrollCount').textContent = simScrollCount;
    }, 1000);
    
    const timingInterval = setInterval(() => {
        simTimingCount++;
        document.getElementById('timingCount').textContent = simTimingCount;
    }, 2000);
    
    // Stop simulation after 10 seconds
    setTimeout(() => {
        clearInterval(mouseInterval);
        clearInterval(keyInterval);
        clearInterval(scrollInterval);
        clearInterval(timingInterval);
    }, 10000);
}

// Human review request
function requestHumanReview() {
    const reviewSection = document.getElementById('reviewRequest');
    reviewSection.style.display = 'block';
    reviewSection.scrollIntoView({ behavior: 'smooth' });
    
    let timeLeft = 300; // 5 minutes in seconds
    const timerElement = document.getElementById('timer');
    const progressBar = document.querySelector('.timer-progress');
    
    const timer = setInterval(() => {
        timeLeft--;
        const minutes = Math.floor(timeLeft / 60);
        const seconds = timeLeft % 60;
        
        timerElement.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        progressBar.style.width = `${((300 - timeLeft) / 300) * 100}%`;
        
        if (timeLeft <= 0) {
            clearInterval(timer);
            timerElement.textContent = "Review Complete!";
        }
    }, 1000);
}

// Auto-refresh for admin page
if (window.location.pathname === '/admin') {
    setInterval(() => {
        location.reload();
    }, 10000); // Refresh every 10 seconds
}