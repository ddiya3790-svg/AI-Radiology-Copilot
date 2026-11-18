// AI Radiology Copilot - corrected frontend integration

const API_URL = "http://127.0.0.1:8000";


// ============================================================
// INITIALIZE APPLICATION
// ============================================================

function initializeApp() {

    // --------------------------------------------------------
    // ELEMENT HELPER
    // --------------------------------------------------------

    const $ = (id) =>
        document.getElementById(id);


    // --------------------------------------------------------
    // MAIN ELEMENTS
    // --------------------------------------------------------

    const mriInput =
        $("mriInput");

    const dropZone =
        $("dropZone");

    const previewContainer =
        $("previewContainer");

    const mriPreview =
        $("mriPreview");

    const fileName =
        $("fileName");

    const fileSize =
        $("fileSize");

    const removeBtn =
        $("removeBtn");

    const analyzeBtn =
        $("analyzeBtn");

    const statusText =
        $("status");

    const resultSection =
        $("resultSection");

    const predictedClass =
        $("predictedClass");

    const confidence =
        $("confidence");

    const confidenceFill =
        $("confidenceFill");

    const probabilityList =
        $("probabilityList");

    const gradcamImage =
        $("gradcamImage");


    // --------------------------------------------------------
    // RAG ELEMENTS
    // --------------------------------------------------------

    const ragQuestion =
        $("ragQuestion");

    const askAiBtn =
        $("askAiBtn");

    const ragLoading =
        $("ragLoading");

    const ragAnswer =
        $("ragAnswer");

    const ragAnswerText =
        $("ragAnswerText");

    const ragSources =
        $("ragSources");

    const ragSourcesList =
        $("ragSourcesList");


    // --------------------------------------------------------
    // REPORT ELEMENTS
    // --------------------------------------------------------

    const generateReportBtn =
        $("generateReportBtn");

    const reportLoading =
        $("reportLoading");

    const aiReport =
        $("aiReport");

    const reportContent =
        $("reportContent");


    // ========================================================
    // APPLICATION STATE
    // ========================================================

    let selectedFile = null;


    window.latestPrediction = null;

    window.latestRAG = {
        answer: "",
        sources: []
    };

    window.latestReport = "";


    // ========================================================
    // UTILITY FUNCTIONS
    // ========================================================


    function escapeHTML(value) {

        return String(
            value ?? ""
        )
        .replace(
            /&/g,
            "&amp;"
        )
        .replace(
            /</g,
            "&lt;"
        )
        .replace(
            />/g,
            "&gt;"
        )
        .replace(
            /"/g,
            "&quot;"
        )
        .replace(
            /'/g,
            "&#039;"
        );
    }


    function showStatus(
        message,
        error = false
    ) {

        if (!statusText) {
            return;
        }

        statusText.textContent =
            message;

        statusText.style.color =
            error
                ? "#ff6878"
                : "#6e879d";
    }


    function formatFileSize(bytes) {

        if (bytes < 1024) {

            return `${bytes} B`;

        }


        if (
            bytes <
            1024 * 1024
        ) {

            return `${(
                bytes / 1024
            ).toFixed(1)} KB`;

        }


        return `${(
            bytes /
            (1024 * 1024)
        ).toFixed(2)} MB`;
    }


    function formatClassName(name) {

        if (
            name === "notumor"
        ) {

            return "No Tumor";
        }


        const text =
            String(
                name || "Unknown"
            );


        return (
            text
                .charAt(0)
                .toUpperCase() +
            text.slice(1)
        );
    }


    function isValidImage(file) {

        if (!file) {
            return false;
        }


        const extension =
            file.name
                .split(".")
                .pop()
                .toLowerCase();


        return (
            [
                "jpg",
                "jpeg",
                "png"
            ].includes(
                extension
            ) &&
            (
                !file.type ||
                [
                    "image/jpeg",
                    "image/png"
                ].includes(
                    file.type
                )
            )
        );
    }


    // ========================================================
    // MARKDOWN FORMATTER
    // ========================================================

    function formatMarkdown(
        markdown,
        headingClass = ""
    ) {

        const lines =
            escapeHTML(
                markdown
            )
            .replace(
                /\r/g,
                ""
            )
            .split("\n");


        const output = [];

        let inList = false;

        let inTable = false;

        let tableHeader = true;


        const inline = (text) => {

            return text.replace(
                /\*\*(.+?)\*\*/g,
                "<strong>$1</strong>"
            );
        };


        const closeList = () => {

            if (inList) {

                output.push(
                    "</ul>"
                );

                inList = false;
            }
        };


        const closeTable = () => {

            if (inTable) {

                output.push(
                    "</table></div>"
                );

                inTable = false;
            }
        };


        for (
            const rawLine of lines
        ) {

            const line =
                rawLine.trim();


            // ------------------------------------------------
            // TABLE SEPARATOR
            // ------------------------------------------------

            if (
                /^\|[\s|:\-]+\|$/.test(
                    line
                )
            ) {

                continue;
            }


            // ------------------------------------------------
            // TABLE
            // ------------------------------------------------

            if (
                line.startsWith("|") &&
                line.endsWith("|")
            ) {

                closeList();


                if (!inTable) {

                    output.push(
                        '<div class="rag-table"><table>'
                    );

                    inTable = true;

                    tableHeader = true;
                }


                const cells =
                    line
                        .slice(1, -1)
                        .split("|");


                const tag =
                    tableHeader
                        ? "th"
                        : "td";


                output.push(
                    `<tr>${cells
                        .map(
                            cell =>
                                `<${tag}>${inline(
                                    cell.trim()
                                )}</${tag}>`
                        )
                        .join("")}</tr>`
                );


                tableHeader = false;

                continue;
            }


            closeTable();


            // ------------------------------------------------
            // EMPTY LINE
            // ------------------------------------------------

            if (
                !line ||
                /^---+$/.test(
                    line
                )
            ) {

                closeList();

                continue;
            }


            // ------------------------------------------------
            // HEADINGS
            // ------------------------------------------------

            const heading =
                line.match(
                    /^#{1,4}\s+(.+)$/
                ) ||
                line.match(
                    /^\*\*(.+)\*\*$/
                );


            if (heading) {

                closeList();


                output.push(
                    `<h3${
                        headingClass
                            ? ` class="${headingClass}"`
                            : ""
                    }>${inline(
                        heading[1]
                    )}</h3>`
                );


                continue;
            }


            // ------------------------------------------------
            // BULLET / NUMBERED LIST
            // ------------------------------------------------

            const bullet =
                line.match(
                    /^(?:[-•*]|\d+\.)\s+(.+)$/
                );


            if (bullet) {

                if (!inList) {

                    output.push(
                        `<ul class="${
                            headingClass
                                ? "report-list"
                                : "rag-list"
                        }">`
                    );

                    inList = true;
                }


                output.push(
                    `<li>${inline(
                        bullet[1]
                    )}</li>`
                );


                continue;
            }


            closeList();


            // ------------------------------------------------
            // NORMAL TEXT
            // ------------------------------------------------

            output.push(
                `<p>${inline(
                    line
                )}</p>`
            );
        }


        closeList();

        closeTable();


        return output.join("\n");
    }


    function formatRAGAnswer(
        markdown
    ) {

        return formatMarkdown(
            markdown
        );
    }


    function formatAIReport(
        markdown
    ) {

        return formatMarkdown(
            markdown,
            "report-section-title"
        );
    }


    // ========================================================
    // RESET REPORT
    // ========================================================

    function resetReport() {

        window.latestReport =
            "";


        if (aiReport) {

            aiReport.style.display =
                "none";
        }


        if (reportContent) {

            reportContent.innerHTML =
                "";
        }


        if (reportLoading) {

            reportLoading.style.display =
                "none";
        }
    }


    // ========================================================
    // RESET RAG
    // ========================================================

    function resetRAG() {

        window.latestRAG = {
            answer: "",
            sources: []
        };


        if (ragAnswer) {

            ragAnswer.style.display =
                "none";
        }


        if (ragAnswerText) {

            ragAnswerText.innerHTML =
                "";
        }


        if (ragSources) {

            ragSources.style.display =
                "none";
        }


        if (ragSourcesList) {

            ragSourcesList.innerHTML =
                "";
        }
    }


    // ========================================================
    // SHOW IMAGE PREVIEW
    // ========================================================

    function showPreview(file) {

        if (!isValidImage(file)) {

            showStatus(
                "Please select a JPG, JPEG or PNG image.",
                true
            );

            return;
        }


        selectedFile =
            file;


        window.latestPrediction =
            null;


        resetRAG();

        resetReport();


        if (resultSection) {

            resultSection.classList.add(
                "hidden"
            );
        }


        if (fileName) {

            fileName.textContent =
                file.name;
        }


        if (fileSize) {

            fileSize.textContent =
                formatFileSize(
                    file.size
                );
        }


        if (previewContainer) {

            previewContainer.classList.remove(
                "hidden"
            );

            previewContainer.style.display =
                "";
        }


        if (mriPreview) {

            const reader =
                new FileReader();


            reader.onload =
                function (event) {

                    mriPreview.src =
                        event.target.result;
                };


            reader.readAsDataURL(
                file
            );
        }


        showStatus(
            "MRI image selected. Ready for analysis."
        );


        console.log(
            "Selected MRI:",
            file.name
        );
    }


    // ========================================================
    // FILE INPUT
    // ========================================================

    if (mriInput) {

        mriInput.addEventListener(
            "change",
            function () {

                const file =
                    mriInput.files?.[0];


                if (file) {

                    showPreview(
                        file
                    );
                }
            }
        );
    }


    // ========================================================
    // CLICK + DRAG/DROP UPLOAD
    // ========================================================

    if (
        dropZone &&
        mriInput
    ) {


        // ----------------------------------------------------
        // CLICK UPLOAD AREA
        // ----------------------------------------------------

        dropZone.addEventListener(
            "click",
            function (event) {

                /*
                 * Do not trigger another file picker when
                 * clicking the remove button.
                 */

                if (
                    event.target ===
                    mriInput
                ) {

                    return;
                }


                if (
                    event.target.closest(
                        "#removeBtn"
                    )
                ) {

                    return;
                }


                /*
                 * Open Windows file picker.
                 */

                console.log(
                    "Upload area clicked - opening file picker"
                );


                mriInput.click();
            }
        );


        // ----------------------------------------------------
        // KEYBOARD ACCESS
        // ----------------------------------------------------

        dropZone.addEventListener(
            "keydown",
            function (event) {

                if (
                    (
                        event.key ===
                        "Enter" ||
                        event.key ===
                        " "
                    ) &&
                    event.target ===
                    dropZone
                ) {

                    event.preventDefault();

                    mriInput.click();
                }
            }
        );


        // ----------------------------------------------------
        // DRAG OVER
        // ----------------------------------------------------

        dropZone.addEventListener(
            "dragover",
            function (event) {

                event.preventDefault();

                dropZone.classList.add(
                    "dragover"
                );
            }
        );


        // ----------------------------------------------------
        // DRAG LEAVE
        // ----------------------------------------------------

        dropZone.addEventListener(
            "dragleave",
            function () {

                dropZone.classList.remove(
                    "dragover"
                );
            }
        );


        // ----------------------------------------------------
        // DROP
        // ----------------------------------------------------

        dropZone.addEventListener(
            "drop",
            function (event) {

                event.preventDefault();


                dropZone.classList.remove(
                    "dragover"
                );


                const file =
                    event.dataTransfer
                        ?.files?.[0];


                if (!file) {

                    return;
                }


                if (
                    !isValidImage(file)
                ) {

                    showStatus(
                        "Please select a JPG, JPEG or PNG image.",
                        true
                    );

                    return;
                }


                /*
                 * Keep the input synchronized with the
                 * dropped file where the browser supports it.
                 */

                try {

                    const transfer =
                        new DataTransfer();


                    transfer.items.add(
                        file
                    );


                    mriInput.files =
                        transfer.files;

                }
                catch (error) {

                    console.warn(
                        "Using dropped file directly:",
                        error
                    );
                }


                showPreview(
                    file
                );
            }
        );
    }


    // ========================================================
    // REMOVE IMAGE
    // ========================================================

    if (removeBtn) {

        removeBtn.addEventListener(
            "click",
            function (event) {

                event.preventDefault();

                event.stopPropagation();


                selectedFile =
                    null;


                if (mriInput) {

                    mriInput.value =
                        "";
                }


                if (mriPreview) {

                    mriPreview.removeAttribute(
                        "src"
                    );
                }


                if (previewContainer) {

                    previewContainer.classList.add(
                        "hidden"
                    );
                }


                if (resultSection) {

                    resultSection.classList.add(
                        "hidden"
                    );
                }


                if (predictedClass) {

                    predictedClass.textContent =
                        "—";
                }


                if (confidence) {

                    confidence.textContent =
                        "—";
                }


                if (confidenceFill) {

                    confidenceFill.style.width =
                        "0%";
                }


                if (probabilityList) {

                    probabilityList.innerHTML =
                        "";
                }


                if (gradcamImage) {

                    gradcamImage.removeAttribute(
                        "src"
                    );
                }


                window.latestPrediction =
                    null;


                if (ragQuestion) {

                    ragQuestion.value =
                        "";
                }


                resetRAG();

                resetReport();


                showStatus(
                    "Ready to analyze your MRI scan."
                );
            }
        );
    }


    // ========================================================
    // DISPLAY PROBABILITIES
    // ========================================================

    function displayProbabilities(
        probabilities = {}
    ) {

        if (!probabilityList) {

            return;
        }


        probabilityList.innerHTML =
            "";


        const colors = {

            glioma:
                "#ff6878",

            meningioma:
                "#ffae52",

            notumor:
                "#27d69b",

            pituitary:
                "#00c6e6"
        };


        Object.entries(
            probabilities
        )
        .sort(
            (a, b) =>
                Number(b[1]) -
                Number(a[1])
        )
        .forEach(
            function (
                [
                    className,
                    probability
                ]
            ) {

                const value =
                    Number(
                        probability
                    );


                const item =
                    document.createElement(
                        "div"
                    );


                item.className =
                    "probability-item";


                const top =
                    document.createElement(
                        "div"
                    );


                top.className =
                    "probability-top";


                const label =
                    document.createElement(
                        "span"
                    );


                label.textContent =
                    formatClassName(
                        className
                    );


                const percentage =
                    document.createElement(
                        "strong"
                    );


                percentage.textContent =
                    `${value.toFixed(2)}%`;


                top.appendChild(
                    label
                );


                top.appendChild(
                    percentage
                );


                const track =
                    document.createElement(
                        "div"
                    );


                track.className =
                    "probability-track";


                const fill =
                    document.createElement(
                        "div"
                    );


                fill.className =
                    "probability-fill";


                fill.style.background =
                    colors[
                        className
                    ] ||
                    "#00c6e6";


                fill.style.width =
                    "0%";


                track.appendChild(
                    fill
                );


                item.appendChild(
                    top
                );


                item.appendChild(
                    track
                );


                probabilityList.appendChild(
                    item
                );


                setTimeout(
                    function () {

                        fill.style.width =
                            `${Math.max(
                                0,
                                Math.min(
                                    value,
                                    100
                                )
                            )}%`;

                    },
                    100
                );
            }
        );
    }


    // ========================================================
    // ANALYZE MRI
    // ========================================================

    if (analyzeBtn) {

        analyzeBtn.addEventListener(
            "click",
            async function (event) {

                event.preventDefault();


                console.log(
                    "Analyze button clicked"
                );


                const file =
                    selectedFile ||
                    mriInput?.files?.[0];


                if (
                    !isValidImage(
                        file
                    )
                ) {

                    showStatus(
                        "Please select a valid MRI image first.",
                        true
                    );

                    return;
                }


                const originalHTML =
                    analyzeBtn.innerHTML;


                analyzeBtn.disabled =
                    true;


                analyzeBtn.innerHTML = `
                    <span class="btn-icon">
                        ◌
                    </span>
                    Analyzing MRI...
                    <span class="btn-arrow">
                        →
                    </span>
                `;


                showStatus(
                    "AI model is analyzing the MRI. Please wait..."
                );


                if (resultSection) {

                    resultSection.classList.add(
                        "hidden"
                    );
                }


                try {

                    const formData =
                        new FormData();


                    formData.append(
                        "file",
                        file
                    );


                    console.log(
                        "Sending MRI to:",
                        `${API_URL}/predict`
                    );


                    const response =
                        await fetch(
                            `${API_URL}/predict`,
                            {
                                method:
                                    "POST",

                                body:
                                    formData
                            }
                        );


                    console.log(
                        "API response status:",
                        response.status
                    );


                    const data =
                        await response.json();


                    console.log(
                        "API response:",
                        data
                    );


                    if (
                        !response.ok ||
                        !data.prediction
                    ) {

                        throw new Error(
                            data.detail ||
                            "Prediction failed."
                        );
                    }


                    const result =
                        data.prediction;


                    const predicted =
                        result.class ||
                        result.predicted_class;


                    const confidenceValue =
                        Number(
                            result.confidence
                        );


                    if (
                        !predicted ||
                        !Number.isFinite(
                            confidenceValue
                        )
                    ) {

                        throw new Error(
                            "The prediction response is missing required fields."
                        );
                    }


                    // ------------------------------------------------
                    // SAVE PREDICTION
                    // ------------------------------------------------

                    window.latestPrediction = {

                        class:
                            predicted,

                        confidence:
                            confidenceValue,

                        probabilities:
                            result.probabilities ||
                            {},

                        gradcamGenerated:
                            Boolean(
                                data.gradcam?.url ||
                                data.gradcam_url ||
                                data.gradcam_generated
                            )
                    };


                    console.log(
                        "Latest prediction saved:",
                        window.latestPrediction
                    );


                    // ------------------------------------------------
                    // DISPLAY PREDICTION
                    // ------------------------------------------------

                    if (
                        predictedClass
                    ) {

                        predictedClass.textContent =
                            formatClassName(
                                predicted
                            );
                    }


                    if (
                        confidence
                    ) {

                        confidence.textContent =
                            `${confidenceValue.toFixed(
                                2
                            )}%`;
                    }


                    // ------------------------------------------------
                    // CONFIDENCE BAR
                    // ------------------------------------------------

                    if (
                        confidenceFill
                    ) {

                        confidenceFill.style.width =
                            "0%";


                        setTimeout(
                            function () {

                                confidenceFill.style.width =
                                    `${Math.max(
                                        0,
                                        Math.min(
                                            confidenceValue,
                                            100
                                        )
                                    )}%`;

                            },
                            100
                        );
                    }


                    // ------------------------------------------------
                    // PROBABILITIES
                    // ------------------------------------------------

                    displayProbabilities(
                        result.probabilities ||
                        {}
                    );


                    // ------------------------------------------------
                    // GRAD-CAM
                    // ------------------------------------------------

                    const gradcamURL =
                        data.gradcam?.url ||
                        data.gradcam_url;


                    if (
                        gradcamImage &&
                        gradcamURL
                    ) {

                        console.log(
                            "Grad-CAM URL:",
                            gradcamURL
                        );


                        const fullURL =
                            gradcamURL.startsWith(
                                "http"
                            )
                                ? gradcamURL
                                : `${API_URL}${gradcamURL}`;


                        gradcamImage.src =
                            `${fullURL}${
                                fullURL.includes(
                                    "?"
                                )
                                    ? "&"
                                    : "?"
                            }t=${Date.now()}`;
                    }


                    // ------------------------------------------------
                    // RESET RAG
                    // ------------------------------------------------

                    resetRAG();


                    // ------------------------------------------------
                    // RESET REPORT
                    // ------------------------------------------------

                    resetReport();


                    // ------------------------------------------------
                    // SHOW RESULTS
                    // ------------------------------------------------

                    if (
                        resultSection
                    ) {

                        resultSection.classList.remove(
                            "hidden"
                        );


                        resultSection.scrollIntoView({
                            behavior:
                                "smooth",

                            block:
                                "start"
                        });
                    }


                    showStatus(
                        "Analysis completed successfully."
                    );


                    console.log(
                        "Prediction:",
                        window.latestPrediction
                    );
                }


                catch (error) {

                    console.error(
                        "Prediction error:",
                        error
                    );


                    showStatus(
                        `Error: ${error.message}`,
                        true
                    );
                }


                finally {

                    analyzeBtn.disabled =
                        false;


                    analyzeBtn.innerHTML =
                        originalHTML;
                }
            }
        );
    }


    // ========================================================
    // RAG MEDICAL ASSISTANT
    // ========================================================

    if (askAiBtn) {

        askAiBtn.addEventListener(
            "click",
            async function (event) {

                event.preventDefault();


                const question =
                    ragQuestion?.value.trim() ||
                    "";


                if (!question) {

                    ragQuestion?.focus();

                    return;
                }


                if (
                    !window.latestPrediction
                ) {

                    alert(
                        "Please analyze an MRI image first."
                    );

                    return;
                }


                if (ragLoading) {

                    ragLoading.style.display =
                        "flex";
                }


                if (ragAnswer) {

                    ragAnswer.style.display =
                        "none";
                }


                if (ragSources) {

                    ragSources.style.display =
                        "none";
                }


                askAiBtn.disabled =
                    true;


                try {

                    console.log(
                        "Sending RAG question:",
                        question
                    );


                    const response =
                        await fetch(
                            `${API_URL}/ask`,
                            {
                                method:
                                    "POST",

                                headers: {
                                    "Content-Type":
                                        "application/json"
                                },

                                body:
                                    JSON.stringify({

                                        predicted_class:
                                            window
                                                .latestPrediction
                                                .class,

                                        confidence:
                                            window
                                                .latestPrediction
                                                .confidence,

                                        question:
                                            question
                                    })
                            }
                        );


                    console.log(
                        "RAG API response status:",
                        response.status
                    );


                    const data =
                        await response.json();


                    console.log(
                        "RAG API response:",
                        data
                    );


                    if (
                        !response.ok ||
                        !data.success
                    ) {

                        throw new Error(
                            data.detail ||
                            data.error ||
                            "RAG request failed."
                        );
                    }


                    // ------------------------------------------------
                    // SAVE RAG
                    // ------------------------------------------------

                    window.latestRAG = {

                        answer:
                            data.answer ||
                            "",

                        sources:
                            data.sources ||
                            []
                    };


                    console.log(
                        "Latest RAG response saved:",
                        window.latestRAG
                    );


                    // ------------------------------------------------
                    // DISPLAY ANSWER
                    // ------------------------------------------------

                    if (
                        ragAnswerText
                    ) {

                        ragAnswerText.innerHTML =
                            formatRAGAnswer(
                                window
                                    .latestRAG
                                    .answer
                            );
                    }


                    if (
                        ragAnswer
                    ) {

                        ragAnswer.style.display =
                            "block";
                    }


                    // ------------------------------------------------
                    // DISPLAY SOURCES
                    // ------------------------------------------------

                    if (
                        ragSourcesList
                    ) {

                        ragSourcesList.innerHTML =
                            "";
                    }


                    const seen =
                        new Set();


                    for (
                        const source
                        of window.latestRAG.sources
                    ) {

                        const name =
                            String(
                                source.source ||
                                "Unknown source"
                            );


                        if (
                            seen.has(
                                name
                            )
                        ) {

                            continue;
                        }


                        seen.add(
                            name
                        );


                        const item =
                            document.createElement(
                                "div"
                            );


                        item.className =
                            "source-item";


                        const nameSpan =
                            document.createElement(
                                "span"
                            );


                        nameSpan.className =
                            "source-name";


                        nameSpan.textContent =
                            name;


                        item.appendChild(
                            nameSpan
                        );


                        if (
                            source.score !=
                            null &&
                            Number.isFinite(
                                Number(
                                    source.score
                                )
                            )
                        ) {

                            const scoreSpan =
                                document.createElement(
                                    "span"
                                );


                            scoreSpan.className =
                                "source-score";


                            scoreSpan.textContent =
                                Number(
                                    source.score
                                ).toFixed(
                                    3
                                );


                            item.appendChild(
                                scoreSpan
                            );
                        }


                        if (
                            ragSourcesList
                        ) {

                            ragSourcesList.appendChild(
                                item
                            );
                        }
                    }


                    if (
                        ragSources
                    ) {

                        ragSources.style.display =
                            seen.size
                                ? "block"
                                : "none";
                    }


                    if (
                        ragAnswer
                    ) {

                        ragAnswer.scrollIntoView({
                            behavior:
                                "smooth",

                            block:
                                "center"
                        });
                    }


                    console.log(
                        "RAG answer displayed."
                    );
                }


                catch (error) {

                    console.error(
                        "RAG error:",
                        error
                    );


                    if (
                        ragAnswerText
                    ) {

                        ragAnswerText.textContent =
                            `Unable to generate an answer: ${error.message}`;
                    }


                    if (
                        ragAnswer
                    ) {

                        ragAnswer.style.display =
                            "block";
                    }
                }


                finally {

                    if (
                        ragLoading
                    ) {

                        ragLoading.style.display =
                            "none";
                    }


                    askAiBtn.disabled =
                        false;
                }
            }
        );
    }


    // ========================================================
    // AI RADIOLOGY REPORT
    // ========================================================

    async function generateAIReport() {

        // ----------------------------------------------------
        // CHECK PREDICTION
        // ----------------------------------------------------

        if (
            !window.latestPrediction
        ) {

            alert(
                "Please analyze an MRI image first."
            );

            return null;
        }


        // ----------------------------------------------------
        // SAVE ORIGINAL BUTTON
        // ----------------------------------------------------

        const originalHTML =
            generateReportBtn
                ?.innerHTML;


        // ----------------------------------------------------
        // LOADING STATE
        // ----------------------------------------------------

        if (
            generateReportBtn
        ) {

            generateReportBtn.disabled =
                true;


            generateReportBtn.textContent =
                "Generating Report...";
        }


        if (
            reportLoading
        ) {

            reportLoading.style.display =
                "flex";
        }


        if (
            aiReport
        ) {

            aiReport.style.display =
                "none";
        }


        try {

            console.log(
                "Generating AI report..."
            );


            // ------------------------------------------------
            // CALL REPORT API
            // ------------------------------------------------

            const response =
                await fetch(
                    `${API_URL}/report`,
                    {
                        method:
                            "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify({

                                predicted_class:
                                    window
                                        .latestPrediction
                                        .class,

                                confidence:
                                    window
                                        .latestPrediction
                                        .confidence,

                                probabilities:
                                    window
                                        .latestPrediction
                                        .probabilities,

                                gradcam_generated:
                                    window
                                        .latestPrediction
                                        .gradcamGenerated,

                                rag_answer:
                                    window
                                        .latestRAG
                                        .answer ||
                                    "",

                                sources:
                                    window
                                        .latestRAG
                                        .sources ||
                                    []
                            })
                    }
                );


            console.log(
                "Report API response status:",
                response.status
            );


            // ------------------------------------------------
            // READ RESPONSE
            // ------------------------------------------------

            const data =
                await response.json();


            console.log(
                "Report API response:",
                data
            );


            // ------------------------------------------------
            // ERROR CHECK
            // ------------------------------------------------

            if (
                !response.ok ||
                !data.success
            ) {

                throw new Error(
                    data.detail ||
                    data.error ||
                    "Report generation failed."
                );
            }


            // ------------------------------------------------
            // GET REPORT TEXT
            // ------------------------------------------------

            let reportText;


            if (
                typeof data.report ===
                "string"
            ) {

                reportText =
                    data.report;

            }
            else if (
                data.report &&
                typeof data.report.report ===
                "string"
            ) {

                reportText =
                    data.report.report;

            }
            else {

                reportText =
                    "";
            }


            // ------------------------------------------------
            // CHECK REPORT
            // ------------------------------------------------

            if (
                typeof reportText !==
                    "string" ||
                !reportText.trim()
            ) {

                throw new Error(
                    "The server returned an empty report."
                );
            }


            // ------------------------------------------------
            // CHECK HTML ELEMENTS
            // ------------------------------------------------

            if (
                !reportContent ||
                !aiReport
            ) {

                throw new Error(
                    "Report display elements are missing from index.html."
                );
            }


            // ------------------------------------------------
            // SAVE REPORT
            // ------------------------------------------------

            window.latestReport =
                reportText;


            // ------------------------------------------------
            // DISPLAY REPORT
            // ------------------------------------------------

            reportContent.innerHTML =
                formatAIReport(
                    reportText
                );


            // ------------------------------------------------
            // MAKE REPORT VISIBLE
            // ------------------------------------------------

            aiReport.classList.remove(
                "hidden"
            );


            aiReport.style.display =
                "block";


            // ------------------------------------------------
            // SCROLL TO REPORT
            // ------------------------------------------------

            aiReport.scrollIntoView({
                behavior:
                    "smooth",

                block:
                    "start"
            });


            console.log(
                "AI report displayed successfully."
            );


            return reportText;
        }


        catch (error) {

            console.error(
                "Report error:",
                error
            );


            alert(
                `Unable to generate AI report: ${error.message}`
            );


            return null;
        }


        finally {

            // ------------------------------------------------
            // HIDE LOADING
            // ------------------------------------------------

            if (
                reportLoading
            ) {

                reportLoading.style.display =
                    "none";
            }


            // ------------------------------------------------
            // RESTORE BUTTON
            // ------------------------------------------------

            if (
                generateReportBtn
            ) {

                generateReportBtn.disabled =
                    false;


                generateReportBtn.innerHTML =
                    originalHTML;
            }
        }
    }


    // ========================================================
    // MAKE REPORT FUNCTION GLOBAL
    // ========================================================

    window.generateAIReport =
        generateAIReport;


    // ========================================================
    // REPORT BUTTON
    // ========================================================

    if (
        generateReportBtn
    ) {

        generateReportBtn.addEventListener(
            "click",
            function (event) {

                event.preventDefault();


                console.log(
                    "Generate AI Report button clicked."
                );


                generateAIReport();
            }
        );
    }


    // ========================================================
    // APPLICATION READY
    // ========================================================

    console.log(
        "AI Radiology Copilot ready: upload, prediction, RAG and report connected."
    );
}


// ============================================================
// START APPLICATION
// ============================================================

if (
    document.readyState ===
    "loading"
) {

    document.addEventListener(
        "DOMContentLoaded",
        initializeApp,
        {
            once: true
        }
    );

}
else {

    initializeApp();
}