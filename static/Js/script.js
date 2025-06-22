document.addEventListener('DOMContentLoaded', function () {
    const glow = document.querySelector('.cursor-glow');

    // Cursor glow effect - optional for all pages
    if (glow) {
        const GLOW_WIDTH = 220;
        const GLOW_HEIGHT = 220;
        document.addEventListener('mousemove', (e) => {
            const x = e.clientX - GLOW_WIDTH / 2;
            const y = e.clientY - GLOW_HEIGHT / 2;
            glow.style.transform = `translate(${x}px, ${y}px)`;
        });
    }

    // === IMAGE UPLOAD PAGE LOGIC (Run only if upload form is found) ===
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const scanButton = document.querySelector('.search-bar button');

    if (uploadForm && fileInput && scanButton) {
        const resultContainer = document.getElementById('resultContainer');
        const errorDisplay = document.getElementById('errorDisplay');
        const resultImage = document.getElementById('resultImage');
        const stampImage = document.getElementById('stampImage');
        const modal = document.getElementById('resultModal');
        const modalContent = modal.querySelector(".modal-content");
        const modalImg = document.getElementById('modalResultImage');
        const MIN_LOADING_TIME = 3000;
        let loadingStartTime = 0;
        let responseData = null;

        uploadForm.addEventListener('submit', handleFormSubmit);
        fileInput.addEventListener('change', handleFileSelect);

        scanButton.disabled = true;
        scanButton.style.opacity = '0.7';
        scanButton.style.cursor = 'not-allowed';

        function handleFileSelect(e) {
            const hasFile = fileInput.files.length > 0;
            scanButton.disabled = !hasFile;
            scanButton.style.opacity = hasFile ? '1' : '0.7';
            scanButton.style.cursor = hasFile ? 'pointer' : 'not-allowed';
        }

        async function handleFormSubmit(e) {
            e.preventDefault();
            hideError();
            hideResult();
            showLoadingOverlay();

            if (!fileInput.files.length) {
                showError('Please select an image first');
                hideLoadingOverlay();
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            scanButton.textContent = "Scanning...";
            scanButton.disabled = true;
            responseData = null;

            try {
                const response = await fetch('/', {
                    method: 'POST',
                    body: formData,
                    headers: { 'Accept': 'application/json' }
                });

                responseData = await response.json();

                if (!response.ok) throw new Error(responseData.error || 'Server error');

                const elapsed = Date.now() - loadingStartTime;
                const remainingTime = Math.max(0, MIN_LOADING_TIME - elapsed);

                setTimeout(() => {
                    hideLoadingOverlay();
                    displayResults(responseData);
                }, remainingTime);

            } catch (error) {
                const elapsed = Date.now() - loadingStartTime;
                const remainingTime = Math.max(0, MIN_LOADING_TIME - elapsed);

                setTimeout(() => {
                    hideLoadingOverlay();
                    showError(error.message);
                    console.error('Error:', error);
                }, remainingTime);
            } finally {
                setTimeout(() => {
                    scanButton.textContent = "Scan Now";
                    scanButton.disabled = false;
                }, MIN_LOADING_TIME);
            }
        }

        function displayResults(data) {
            const isMorphed = data.status === 'MORPHED';
            document.getElementById('warningNote').style.display = 'block';

            document.getElementById("modalTitle").textContent = isMorphed
                ? "Morphed Image Detected"
                : "Original Image Verified";

            const confidencePercent = parseFloat(data.confidence).toFixed(2);
            /*document.getElementById('modalConfidenceValue').textContent = `${confidencePercent}%`;

            const fill = document.getElementById('modalConfidenceFill');
            fill.style.width = `${confidencePercent}%`;
            fill.style.backgroundColor = isMorphed ? '#f44336' : '#4caf50';*/

            const uploadedImage = document.getElementById('uploadedImage');
            const stampImage = document.getElementById('stampImage');
            const resultSection = document.getElementById('resultSection');

            uploadedImage.src = data.image_url;
            stampImage.src = data.stamp_url;

            resultSection.style.display = 'flex';

            document.getElementById('modalResultDescription').textContent = isMorphed
                ? "MorphShield has detected signs of morphing and tampering in the uploaded image. The image is flagged as inauthentic due to structural and visual inconsistencies."
                : "MorphShield confirms the uploaded image is authentic, with no signs of morphing or tampering. The image passes all forensic and structural integrity checks.";

            modalContent.classList.remove("morphed", "original");
            modalContent.classList.add(isMorphed ? "morphed" : "original");

            modal.style.display = 'block';

            document.getElementById('closeModal').onclick = () => {
                modal.style.display = 'none';
                modalImg.src = '';
            };

            window.onclick = event => {
                if (event.target === modal) {
                    modal.style.display = 'none';
                    modalImg.src = '';
                }
            };
        }

        function showError(message) {
            errorDisplay.textContent = message;
            errorDisplay.style.display = 'block';
            errorDisplay.style.opacity = '1';
        }

        function hideError() {
            errorDisplay.style.opacity = '0';
            setTimeout(() => errorDisplay.style.display = 'none', 300);
        }

        function hideResult() {
            resultContainer.style.opacity = '0';
            resultImage.style.display = 'none';
            setTimeout(() => {
                resultContainer.style.display = 'none';
            }, 300);
        }

        function showLoadingOverlay() {
            loadingStartTime = Date.now();
            document.getElementById('loadingOverlay').style.display = 'flex';
        }

        function hideLoadingOverlay() {
            const overlay = document.getElementById('loadingOverlay');
            overlay.classList.add('hiding');
            const elapsed = Date.now() - loadingStartTime;
            const remainingTime = Math.max(0, MIN_LOADING_TIME - elapsed);

            setTimeout(() => {
                overlay.style.display = 'none';
                overlay.classList.remove('hiding');
            }, remainingTime);
        }
    }
});
