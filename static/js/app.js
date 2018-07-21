$('document').ready(function () {
    $('.loading').hide();
    
    $('#send-params').submit(function () {
        $('#btn-submit-params').attr('disabled', true);
        $('#loading').fadeIn(300);
        $('#loading').css('visibility', 'visible');
    });
});