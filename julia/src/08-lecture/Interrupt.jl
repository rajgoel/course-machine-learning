"""
Global reference to track interrupt status.
"""
QUIT = Ref(false)

"""
    enable_interrupt()

Start monitoring for ENTER key press to interrupt training.
Press ENTER to interrupt training gracefully.
"""
function enable_interrupt()
    QUIT[] = false
    println("Press ENTER to interrupt training...")
    
    @async begin
        while !QUIT[]
            try
                readline()  # Wait for ENTER
                QUIT[] = true
                println("ENTER pressed - interrupting...")
                break
            catch
                break
            end
        end
    end
end
