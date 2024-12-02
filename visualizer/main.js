document.addEventListener("DOMContentLoaded", () => {
    const canvas = document.getElementById("trackingCanvas");
    const ctx = canvas.getContext("2d");

    const prevButton = document.getElementById("prevFrame");
    const nextButton = document.getElementById("nextFrame");
    const frameSlider = document.getElementById("frameSlider");

    let isPlaying = false;
    let playbackInterval = null;

    let selectedIds = new Set();
    let trackingData = [];
    const framePaths = []; // Array to hold frame image paths
    let currentFrameIndex = 0;
    const classColors = {
        0: "red",
        1: "blue",
        2: "green",
        3: "yellow",
    };


    document.getElementById("uploadJson").addEventListener("change", event => {
        const file = event.target.files[0];
        if (!file) {
            alert("No file selected.");
            return;
        }

        const reader = new FileReader();
        reader.onload = function (e) {
            try {
                trackingData = JSON.parse(e.target.result);
                normalizeTrackIds(); // Ensure no gaps in track IDs

                const allIds = new Set();
                trackingData.forEach(frame => {
                    frame.objects.forEach(obj => allIds.add(obj.track_id));
                });
        
                // Populate the dropdown
                const idFilter = document.getElementById("idFilter");
                allIds.forEach(id => {
                    const option = document.createElement("option");
                    option.value = id;
                    option.textContent = `ID ${id}`;
                    idFilter.appendChild(option);
                    selectedIds.add(id); // Initially, all IDs are selected
                });
        
                frameSlider.max = trackingData.length - 1;

                alert("JSON file loaded successfully!");
                updatePlot(); // Refresh visualization
            } catch (err) {
                alert("Failed to load JSON file: " + err.message);
            }
        };
        reader.readAsText(file);

     
    });


    document.getElementById("playButton").addEventListener("click", () => {
        if (isPlaying) return;

        isPlaying = true;
        document.getElementById("playButton").disabled = true;
        document.getElementById("pauseButton").disabled = false;

        // Start playback based on the FPS slider value
        const fps = parseInt(document.getElementById("fpsSlider").value, 10);
        const interval = 1000 / fps; // Convert FPS to interval in milliseconds

        playbackInterval = setInterval(() => {
            nextFrame(); // Advance to the next frame
        }, interval);
    });

    document.getElementById("pauseButton").addEventListener("click", () => {
        isPlaying = false;
        document.getElementById("playButton").disabled = false;
        document.getElementById("pauseButton").disabled = true;

        // Stop playback
        clearInterval(playbackInterval);
    });

    document.getElementById("fpsSlider").addEventListener("input", () => {
        const fps = parseInt(document.getElementById("fpsSlider").value, 10);
        document.getElementById("fpsValue").textContent = fps;

        if (isPlaying) {
            // Restart playback with the new FPS value
            clearInterval(playbackInterval);
            const interval = 1000 / fps;
            playbackInterval = setInterval(() => {
                nextFrame(); // Advance to the next frame
            }, interval);
        }
    });

    document.getElementById("mergeSelectedIds").addEventListener("click", () => {
        if (selectedIds.size < 2) {
            alert("Please select at least two track IDs to merge.");
            return;
        }

        // Convert selectedIds to an array and sort to find the lowest ID
        const sortedIds = Array.from(selectedIds).map(id => parseInt(id, 10)).sort((a, b) => a - b);
        const lowestId = sortedIds[0];

        // Merge all higher IDs into the lowest ID across all frames
        trackingData.forEach(frame => {
            frame.objects.forEach(obj => {
                if (selectedIds.has(String(obj.track_id)) && obj.track_id !== lowestId) {
                    obj.track_id = lowestId;
                }
            });
        });

        // Update the plot to reflect changes
        updatePlot();
        alert(`Merged selected IDs into track ID ${lowestId} across all frames.`);
    });

    document.getElementById("selectAll").addEventListener("click", () => {
        const idFilter = document.getElementById("idFilter");
        selectedIds = new Set([...idFilter.options].map(option => option.value));
        [...idFilter.options].forEach(option => option.selected = true);
        updatePlot();
    });

    document.getElementById("selectNone").addEventListener("click", () => {
        selectedIds.clear();
        const idFilter = document.getElementById("idFilter");
        [...idFilter.options].forEach(option => option.selected = false);
        updatePlot();
    });

    document.getElementById("idFilter").addEventListener("change", (event) => {
        const selectedOptions = [...event.target.options].filter(option => option.selected);
        selectedIds = new Set(selectedOptions.map(option => option.value));
        updatePlot();
    });

    document.getElementById("setTrackClass").addEventListener("click", () => {
        const newClassId = parseInt(document.getElementById("newClassId").value, 10);
        if (isNaN(newClassId) || newClassId < 0 || newClassId > 3) {
            alert("Please enter a valid class ID between 0 and 3.");
            return;
        }

        // Modify all frames' objects to set the new class for selected IDs
        trackingData.forEach(frame => {
            frame.objects.forEach(obj => {
                if (selectedIds.has(String(obj.track_id))) {
                    obj.class_id = newClassId;
                }
            });
        });

        // Update the plot to reflect changes
        updatePlot();
        alert(`Selected tracks across all frames set to class ${newClassId}.`);
    });


    document.getElementById("swapTrackIds").addEventListener("click", () => {
        if (selectedIds.size !== 2) {
            alert("Please select exactly two track IDs to swap.");
            return;
        }

        const [id1, id2] = Array.from(selectedIds).map(id => parseInt(id, 10));

        // Swap the IDs in all frames starting from the current frame
        for (let i = currentFrameIndex; i < trackingData.length; i++) {
            trackingData[i].objects.forEach(obj => {
                if (obj.track_id === id1) {
                    obj.track_id = id2;
                } else if (obj.track_id === id2) {
                    obj.track_id = id1;
                }
            });
        }

        // Update the plot to reflect changes
        updatePlot();
        alert(`Swapped track IDs ${id1} and ${id2} from frame ${currentFrameIndex + 1} onward.`);
    });

    document.getElementById("saveJson").addEventListener("click", () => {
        if (trackingData.length === 0) {
            alert("No data to save. Please upload and modify a JSON file first.");
            return;
        }

        const dataStr = JSON.stringify(trackingData, null, 2);
        const blob = new Blob([dataStr], { type: "application/json" });

        // Create a temporary download link
        const a = document.createElement("a");
        const url = URL.createObjectURL(blob);
        a.href = url;

        // Trigger file save dialog
        a.download = prompt("Enter a file name (with .json extension):", "modified_tracking_data.json") || "modified_tracking_data.json";
        a.click();

        // Clean up
        URL.revokeObjectURL(url);
        alert("File saved successfully!");
    });


    document.getElementById("removeSelectedIds").addEventListener("click", () => {
        if (selectedIds.size === 0) {
            alert("No IDs selected to remove.");
            return;
        }

        // Remove objects with selected track IDs from all frames
        trackingData.forEach(frame => {
            frame.objects = frame.objects.filter(obj => !selectedIds.has(String(obj.track_id)));
        });

        // Update the plot to reflect changes
        updatePlot();
        alert("Selected track IDs have been removed from all frames.");
    });

    function updateBboxSliders(bbox) {
        document.getElementById("bboxLeft").value = bbox[0];
        document.getElementById("bboxTop").value = bbox[1];
        document.getElementById("bboxRight").value = bbox[2];
        document.getElementById("bboxBottom").value = bbox[3];

        document.getElementById("bboxLeftValue").textContent = bbox[0];
        document.getElementById("bboxTopValue").textContent = bbox[1];
        document.getElementById("bboxRightValue").textContent = bbox[2];
        document.getElementById("bboxBottomValue").textContent = bbox[3];
    }

    function updateCenterFromBbox() {
        const left = parseFloat(document.getElementById("bboxLeft").value);
        const top = parseFloat(document.getElementById("bboxTop").value);
        const right = parseFloat(document.getElementById("bboxRight").value);
        const bottom = parseFloat(document.getElementById("bboxBottom").value);

        const centerX = (left + right) / 2;
        const centerY = (top + bottom) / 2;

        document.getElementById("centerX").value = centerX;
        document.getElementById("centerY").value = centerY;
        document.getElementById("centerXValue").textContent = centerX;
        document.getElementById("centerYValue").textContent = centerY;

        const selectedId = parseInt(Array.from(selectedIds)[0], 10);
        const frameInfo = trackingData[currentFrameIndex];
        frameInfo.objects.forEach(obj => {
            if (obj.track_id === selectedId) {
                obj.center = [centerX, centerY];
            }
        });
    }

    function updateBboxFromCenter() {
        const centerX = parseFloat(document.getElementById("centerX").value);
        const centerY = parseFloat(document.getElementById("centerY").value);
        const left = parseFloat(document.getElementById("bboxLeft").value);
        const top = parseFloat(document.getElementById("bboxTop").value);
        const width = parseFloat(document.getElementById("bboxRight").value) - left;
        const height = parseFloat(document.getElementById("bboxBottom").value) - top;

        const newLeft = centerX - width / 2;
        const newRight = centerX + width / 2;
        const newTop = centerY - height / 2;
        const newBottom = centerY + height / 2;

        document.getElementById("bboxLeft").value = newLeft;
        document.getElementById("bboxRight").value = newRight;
        document.getElementById("bboxTop").value = newTop;
        document.getElementById("bboxBottom").value = newBottom;

        document.getElementById("bboxLeftValue").textContent = newLeft;
        document.getElementById("bboxRightValue").textContent = newRight;
        document.getElementById("bboxTopValue").textContent = newTop;
        document.getElementById("bboxBottomValue").textContent = newBottom;

        const selectedId = parseInt(Array.from(selectedIds)[0], 10);
        const frameInfo = trackingData[currentFrameIndex];
        frameInfo.objects.forEach(obj => {
            if (obj.track_id === selectedId) {
                obj.center = [centerX, centerY];
            }
        });
    }


    function applyBboxChanges() {
        if (selectedIds.size !== 1) {
            alert("Please select exactly one track ID for bounding box operations.");
            return;
        }

        const selectedId = parseInt(Array.from(selectedIds)[0], 10);

        const left = parseFloat(document.getElementById("bboxLeft").value);
        const top = parseFloat(document.getElementById("bboxTop").value);
        const right = parseFloat(document.getElementById("bboxRight").value);
        const bottom = parseFloat(document.getElementById("bboxBottom").value);

        const frameInfo = trackingData[currentFrameIndex];
        frameInfo.objects.forEach(obj => {
            if (obj.track_id === selectedId) {
                obj.bbox = [left, top, right, bottom];
            }
        });

        updatePlot();
        //alert(`Bounding box updated for track ID ${selectedId} in the current frame.`);
    }

    document.getElementById("getCurrentBbox").addEventListener("click", () => {
        if (selectedIds.size !== 1) {
            alert("Please select exactly one track ID to retrieve bounding box.");
            return;
        }

        const selectedId = parseInt(Array.from(selectedIds)[0], 10);
        const frameInfo = trackingData[currentFrameIndex];
        const obj = frameInfo.objects.find(obj => obj.track_id === selectedId);

        if (!obj) {
            alert(`Track ID ${selectedId} not found in the current frame.`);
            return;
        }

        const bbox = obj.bbox;
        updateBboxSliders(bbox);
        updateCenterFromBbox();
        alert(`Bounding box retrieved for track ID ${selectedId}.`);
    });


    document.getElementById("bboxLeft").addEventListener("input", () => {
        applyBboxChanges();
        updateCenterFromBbox();
    });

    document.getElementById("bboxTop").addEventListener("input", () => {
        applyBboxChanges();
        updateCenterFromBbox();
    });

    document.getElementById("bboxRight").addEventListener("input", () => {
        applyBboxChanges();
        updateCenterFromBbox();
    });

    document.getElementById("bboxBottom").addEventListener("input", () => {
        applyBboxChanges();
        updateCenterFromBbox();
    });

    document.getElementById("centerX").addEventListener("input", () => {
        updateBboxFromCenter();
        applyBboxChanges();
    });

    document.getElementById("centerY").addEventListener("input", () => {
        updateBboxFromCenter();
        applyBboxChanges();
    });


    function copyFromFrame(offset) {
        if (selectedIds.size !== 1) {
            alert("Please select exactly one track ID for bounding box operations.");
            return;
        }

        const selectedId = parseInt(Array.from(selectedIds)[0], 10);
        const targetFrameIndex = currentFrameIndex + offset;

        if (targetFrameIndex < 0 || targetFrameIndex >= trackingData.length) {
            alert("Cannot copy bounding box: No next/previous frame available.");
            return;
        }

        const targetFrame = trackingData[targetFrameIndex];
        const currentFrame = trackingData[currentFrameIndex];

        // Find or create the target object in the target frame
        let targetObj = targetFrame.objects.find(obj => obj.track_id === selectedId);
        if (!targetObj) {
            targetObj = {
                track_id: selectedId,
                bbox: [0, 0, 0, 0], // Default bounding box if not found
                center: [0, 0], // Default center if not found
                class_id: 0, // Default class ID
                confidence: 1.0, // Default confidence
            };
            targetFrame.objects.push(targetObj);
        }

        // Calculate center if the targetObj center is missing
        if (!targetObj.center || targetObj.center.length !== 2) {
            const [left, top, right, bottom] = targetObj.bbox;
            targetObj.center = [(left + right) / 2, (top + bottom) / 2];
        }

        // Find or create the current object in the current frame
        let currentObj = currentFrame.objects.find(obj => obj.track_id === selectedId);
        if (!currentObj) {
            currentObj = {
                track_id: selectedId,
                bbox: [...targetObj.bbox], // Copy the target bounding box
                center: [...targetObj.center], // Copy the target center
                class_id: targetObj.class_id, // Copy the class ID
                confidence: 1.0 // Default confidence
            };
            currentFrame.objects.push(currentObj);
        } else {
            // Update the bounding box and center if the object already exists
            currentObj.bbox = [...targetObj.bbox];
            currentObj.center = [...targetObj.center];
            currentObj.confidence = 1.0 // Default confidence
        }

        // Update the sliders and visualization
        updateBboxSliders(currentObj.bbox);
        updateCenterFromBbox();
        updatePlot();
        alert(`Bounding box and center copied from frame ${targetFrameIndex + 1} to current frame.`);
    }

    document.getElementById("copyFromFrame").addEventListener("click", () => {
        const frameInput = parseInt(document.getElementById("copyFromFrameInput").value, 10);
        if (isNaN(frameInput) || frameInput < 1 || frameInput > trackingData.length) {
            alert("Please enter a valid frame number.");
            return;
        }

        const targetFrameIndex = frameInput - 1; // Convert to 0-based index
        const offset = targetFrameIndex - currentFrameIndex; // Calculate offset
        copyFromFrame(offset);
    });


    function normalizeTrackIds() {
        const idMapping = new Map();
        let newId = 1;

        // Create a mapping of old IDs to new sequential IDs
        trackingData.forEach(frame => {
            frame.objects.forEach(obj => {
                if (!idMapping.has(obj.track_id)) {
                    idMapping.set(obj.track_id, newId++);
                }
            });
        });

        // Update all track IDs in the data to use the new mapping
        trackingData.forEach(frame => {
            frame.objects.forEach(obj => {
                obj.track_id = idMapping.get(obj.track_id);
            });
        });

    }




    async function loadFrames() {
        const frameCount = 60; // Adjust based on the number of frames extracted
        for (let i = 1; i <= frameCount; i++) {
            framePaths.push(`output_frames/frame_${String(i).padStart(4, "0")}.png`);
        }
    }

    function updateJsonViewer(frameInfo, filteredObjects) {
        const jsonViewer = document.getElementById("jsonViewer");

        const jsonData = {
            frameIndex: currentFrameIndex,
            selectedObjects: filteredObjects,
        };

        jsonViewer.value = JSON.stringify(jsonData, null, 2);
    }


    function updatePlot() {
        const frameInfo = trackingData[currentFrameIndex];

        const frameImg = new Image();
        frameImg.src = framePaths[currentFrameIndex];

        frameImg.onload = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw frame
            ctx.drawImage(frameImg, 0, 0, canvas.width, canvas.height);

            // Video to canvas scaling factors
            const scaleX = canvas.width / frameImg.width;
            const scaleY = canvas.height / frameImg.height;

            // Filter objects by selected IDs
            const filteredObjects = frameInfo.objects.filter(obj => selectedIds.has(String(obj.track_id)));

            // Draw objects
            filteredObjects.forEach(obj => {
                if (obj.confidence < 0.1) return;

                const { class_id, center, track_id, bbox } = obj;
                const color = classColors[class_id] || "white";

                // Transform coordinates to canvas space
                const centerX = center[0] * scaleX;
                const centerY = center[1] * scaleY;

                const bboxX = bbox[0] * scaleX;
                const bboxY = bbox[1] * scaleY;
                const bboxWidth = (bbox[2] - bbox[0]) * scaleX;
                const bboxHeight = (bbox[3] - bbox[1]) * scaleY;

                // Draw bounding box
                ctx.beginPath();
                ctx.rect(bboxX, bboxY, bboxWidth, bboxHeight);
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.stroke();

                // Draw center point
                ctx.beginPath();
                ctx.arc(centerX, centerY, 5, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.fill();

                // Annotate track ID
                ctx.fillStyle = "white";
                ctx.font = "12px Arial";
                ctx.fillText(`ID: ${track_id}`, centerX + 10, centerY - 10);
            });

            // Display frame index
            ctx.fillStyle = "white";
            ctx.font = "16px Arial";
            ctx.fillText(`Frame: ${currentFrameIndex + 1}/${trackingData.length}`, 10, 20);

            // Update JSON viewer
            updateJsonViewer(frameInfo, filteredObjects);
        };
    }


    function nextFrame() {
        currentFrameIndex = (currentFrameIndex + 1) % framePaths.length;
        frameSlider.value = currentFrameIndex;
        updatePlot();
    }

    function prevFrame() {
        currentFrameIndex = (currentFrameIndex - 1 + framePaths.length) % framePaths.length;
        frameSlider.value = currentFrameIndex;
        updatePlot();
    }

    frameSlider.addEventListener("input", () => {
        currentFrameIndex = parseInt(frameSlider.value);
        updatePlot();
    });

    nextButton.addEventListener("click", nextFrame);
    prevButton.addEventListener("click", prevFrame);

    // Initialize
    
    loadFrames();
});
