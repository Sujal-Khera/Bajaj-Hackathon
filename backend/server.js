const express = require('express');
const bodyParser = require('body-parser');
const multer = require('multer');
const cors = require('cors');
const app = express();
const port = 3000;

app.set('view engine', 'ejs'); // Set EJS as the view engine
app.use(cors()); // Enable CORS for all routes
app.use(express.static('public')); // Serve static files (e.g., CSS, JS) from 'public'
app.use(bodyParser.json());

const storage = multer.memoryStorage();
const upload = multer({ 
    storage: storage, 
    limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
    fileFilter: (req, file, cb) => {
        const allowedTypes = ['.pdf', '.txt', '.docx', '.doc', '.md'];
        const fileExtension = '.' + file.originalname.split('.').pop().toLowerCase();
        if (allowedTypes.includes(fileExtension)) {
            cb(null, true);
        } else {
            cb(new Error('Unsupported file type. Use PDF, TXT, DOCX, DOC, or MD.'), false);
        }
    }
}).single('document');

let documents = []; // In-memory storage for uploaded documents
let documentNames = []; // To store original filenames

app.get('/', (req, res) => {
    res.render('index'); // Render the EJS template (looks for 'views/index.ejs')
});

app.post('/upload', (req, res) => {
    upload(req, res, (err) => {
        if (err) {
            return res.status(400).json({ message: err.message });
        }
        if (req.file) {
            const fileContent = req.file.buffer.toString('utf-8'); // Read as text
            documents.push(fileContent);
            documentNames.push(req.file.originalname);
            res.json({ message: 'Document uploaded successfully', filename: req.file.originalname });
        } else {
            res.status(400).json({ message: 'No file uploaded' });
        }
    });
});

app.post('/query', (req, res) => {
    const { prompt } = req.body;
    let response = 'No relevant information found in the document.';
    
    if (prompt && documents.length > 0) {
        documents.forEach((doc, index) => {
            const lowerDoc = doc.toLowerCase();
            const lowerPrompt = prompt.toLowerCase();
            const indexOfPrompt = lowerDoc.indexOf(lowerPrompt);
            if (indexOfPrompt !== -1) {
                const contextStart = Math.max(0, indexOfPrompt - 50);
                const contextEnd = Math.min(doc.length, indexOfPrompt + prompt.length + 50);
                response = `Found in document "${documentNames[index]}": ...${doc.substring(contextStart, contextEnd)}...`;
            }
        });
        if (prompt.toLowerCase().includes('summary')) {
            response = `Summary: The document "${documentNames[0]}" contains ${documents[0].length} characters.`;
        }
    }
    res.json({ response });
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});