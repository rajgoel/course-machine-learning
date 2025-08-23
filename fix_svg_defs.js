    
function fix_svg_defs( svgObject, prefix ) {
  function guid() {
    function s4() {
      return Math.floor((1 + Math.random()) * 0x10000)
        .toString(16)
        .substring(1);
      }
    return s4() + s4() + '-' + s4() + '-' + s4() + '-' + s4() + '-' + s4() + s4() + s4();
  }

  if ( !prefix) {
    prefix = guid();
  }

  // Select all elements inside <defs>
  const defs = svgObject.querySelectorAll("defs *");
  // Loop through each <defs> element
  defs.forEach((def, index) => {
    const oldId = def.getAttribute("id");
    if (oldId) {
      const newId = prefix + '-' + oldId;  // Create a new ID
        
      // Update the <defs> element's id
      def.setAttribute("id", newId);
        
      // Find and update all references to this old ID in the SVG
       
      // Update each clip-paths using `url(#oldId)`)
      svgObject.querySelectorAll(`[clip-path="url('#${oldId}')"]`).forEach(element => {
        element.setAttribute("clip-path", `url('#${newId}')`);
      });
      
      // Add other defines as needed

//      console.log(`Updated ID from ${oldId} to ${newId}`);
    }
  });
}
