CUSTOM_CSS = """
<style>
body, html {
    height: 100%;
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background-color: #808000;
    color: #000000;
}
.gradio-container {
    background-color: rgba(255, 255, 255, 0.8);
    padding: 30px;
    border-radius: 15px;
    max-width: 900px;
    margin: auto;
    box-shadow: 0 0 15px rgba(0,0,0,0.2);
}
#logo {
    position: absolute;
    top: 20px;
    right: 20px;
    width: 80px;
    height: auto;
    z-index: 1000;
}
button {
    background-color: #009a4d;
    border: none;
    color: black;
    padding: 12px 20px;
    text-align: center;
    font-size: 16px;
    border-radius: 10px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
button:hover {
    background-color: #8fbc8f;
}
.tab-nav button[aria-selected="true"] {
    background-color: #8fbc8f !important;
    color: black !important;
    font-weight: bold;
}
h1, h2, h3, h4 {
    color: #000000;
}
input, textarea {
    border-radius: 8px;
    border: 1px solid #ccc;
    padding: 10px;
    font-size: 16px;
    width: 100%;
    box-sizing: border-box;
    color: #000000;
    background-color: #ffffff;
}
</style>
"""
