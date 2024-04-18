function copyText() {
    // Get the div element
    var divElement = document.getElementById("bibtex-textarea");
  
    // Create a range object
    var range = document.createRange();
  
    // Select the contents of the div element
    range.selectNode(divElement);
  
    // Add the range to the user's selection
    window.getSelection().addRange(range);
  
    // Copy the selected text to the clipboard
    document.execCommand("copy");
  
  }